import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from multitask_sparse_parity import generate_multitask_sparse_parity_dataset
from sgld_utils import (
    SGLDConfig, 
    run_llc_estimation,
)
from utils import to_json_friendly_tree, running_mean
import jax
import jax.numpy as jnp
from typing import Sequence, Callable, Dict, Tuple, Optional
import optax
import numpy as np
import haiku as hk


from sacred import Experiment
# Create a new experiment
ex = Experiment('multitask_sparse_parity')



def data_generator(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    batch_size: int,
    key: jax.random.PRNGKey, 
):
    dataset_size = len(features)
    indices = jnp.arange(dataset_size)
    
    while True:
        key, subkey = jax.random.split(key)
        permuted_indices = jax.random.permutation(subkey, indices)
        
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_indices = permuted_indices[start:end]
            yield features[batch_indices], labels[batch_indices]



class MLP(hk.Module):
    def __init__(
        self,
        hidden_sizes: Sequence[int],
        out_size: int,
        activation: Callable = jax.nn.relu
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for hidden_size in self.hidden_sizes:
            x = hk.Linear(hidden_size)(x)
            x = self.activation(x)
        return hk.Linear(self.out_size)(x)

def create_model(
    hidden_sizes: Sequence[int],
    out_size: int
) -> hk.Transformed:
    def _model_fn(x):
        model = MLP(hidden_sizes, out_size)
        return model(x)
    
    return hk.transform(_model_fn)


@ex.config  
def config():
    expt_name = f"dev"
    data_config = {
        'n_tasks': 30,
        'n_taskbits': 50,
        'n_subset_size': 3,
        'num_training_samples': 100000,
        'num_test_samples': 20000,
        'alpha': 0.4
    }

    model_config = {
        "model_type": "mlp",
        'hidden_sizes': [1024, 1024],
    }

    training_config = {
        "optim": "adam",
        'learning_rate': 1e-3,
        "momentum": 0.9,
        'batch_size': 256,
        'num_steps': 30001,
    }

    sgld_config = {
        'epsilon': 5e-6,
        'gamma': 1.0,
        'num_steps': 1000,
        "num_chains": 1, 
        "batch_size": 256
    }
    early_stopping_epsilon = 1e-8
    max_num_stages = 15

    do_llc_estimation = True
    logging_period = 500


    seed = 0
    dataset_seed = 0



@ex.automain
def run_experiment(
    _run, 
    expt_name,
    data_config,
    model_config,
    training_config,
    sgld_config,
    early_stopping_epsilon,
    max_num_stages,
    do_llc_estimation,
    logging_period,
    seed,
    dataset_seed
):
    # Set random seed
    rngkey = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    
    # Unpack config
    n_tasks, n_taskbits, n_subset_size = data_config['n_tasks'], data_config['n_taskbits'], data_config['n_subset_size']
    num_training_samples = data_config['num_training_samples']
    num_test_samples = data_config['num_test_samples']
    alpha = data_config['alpha']


    hidden_sizes = model_config['hidden_sizes']
    batch_size = training_config['batch_size']
    num_steps = training_config['num_steps']
    learning_rate = training_config['learning_rate']


    sgld_config = SGLDConfig(**sgld_config)
    loss_trace_minibatch = True
    burn_in_prop = 0.9
    itemp = 1.0 / jnp.log(num_training_samples)

    ########################################################
    # Generate dataset
    ########################################################
    if dataset_seed is not None:
        dataset_key = jax.random.PRNGKey(dataset_seed)
    else:
        rngkey, dataset_key = jax.random.split(rngkey)
    data_key, model_key, split_key, gen_key = jax.random.split(dataset_key, 4)
    num_total_samples = num_training_samples + num_test_samples

    features, labels, subtasks = generate_multitask_sparse_parity_dataset(
        data_key,
        n_tasks,
        n_taskbits,
        n_subset_size,
        num_total_samples, 
        alpha=alpha
    )
    labels = jax.nn.one_hot(labels, 2)
    print(labels.shape)

    # Split into train and test sets
    indices = jnp.arange(num_total_samples)
    train_indices = jax.random.choice(split_key, indices, shape=(num_training_samples,), replace=False)
    test_indices = jnp.setdiff1d(indices, train_indices)

    train_features, train_labels = features[train_indices], labels[train_indices]
    test_features, test_labels = features[test_indices], labels[test_indices]

    ########################################################
    # Create model
    ########################################################
    input_size = n_tasks + n_taskbits
    output_size = 2
    model = create_model(hidden_sizes, output_size)
    param_init = model.init(model_key, jnp.zeros((input_size,)))

    num_parameters = sum(np.prod(p.shape) for p in jax.tree_util.tree_flatten(param_init)[0])
    print(f"Number of parameters: {num_parameters}")


    ########################################################
    # Training set up
    ########################################################
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(param_init)

    @jax.jit
    def loss_fn(
        params: hk.Params,
        x: jnp.ndarray,
        y: jnp.ndarray
    ) -> jnp.ndarray:
        # pred = jax.vmap(model.apply, in_axes=(None, None, 0))(params, None, x)
        logits = model.apply(params, None, x)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

    @jax.jit
    def make_step(
        params: hk.Params,
        x: jnp.ndarray,
        y: jnp.ndarray,
        opt_state,
    ):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    @jax.jit
    def compute_accuracy(
        params: hk.Params,
        x: jnp.ndarray,
        y: jnp.ndarray
    ) -> float:
        logits = model.apply(params, None, x)
        # logits = jax.vmap(model.apply, in_axes=(None, None, 0))(params, None, x)
        preds = jax.nn.sigmoid(logits) > 0.5
        return jnp.mean(preds == y)


    def compute_task_losses(
        params: hk.Params,
        features: jnp.ndarray,
        labels: jnp.ndarray,
        n_tasks: int
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        task_losses = {}
        task_errors = {}
        
        for task_idx in range(n_tasks):
            task_mask = features[:, task_idx] == 1
            task_features = features[task_mask]
            task_labels = labels[task_mask]
            loss = loss_fn(params, task_features, task_labels)
            error = 1 - compute_accuracy(params, task_features, task_labels)
            task_losses[task_idx] = float(loss)
            task_errors[task_idx] = float(error)
        return task_losses, task_errors
    
    ########################################################
    # Training loop
    ########################################################

    # Create data generator
    train_gen = data_generator(train_features, train_labels, batch_size, gen_key)

    # Training loop
    _run.info["records"] = []
    param = param_init
    step = 0

    while step < num_steps:
        x_batch, y_batch = next(train_gen)
        loss, param, opt_state = make_step(param, x_batch, y_batch, opt_state)
        
        if step % logging_period == 0:
            test_acc = compute_accuracy(param, test_features, test_labels)
            test_loss = loss_fn(param, test_features, test_labels)
            batch_acc = compute_accuracy(param, x_batch, y_batch)
            
            # Compute task-wise test losses
            task_losses, task_errors = compute_task_losses(param, test_features, test_labels, n_tasks)
            rec = {
                "step": step,
                "loss": float(loss),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "batch_acc": float(batch_acc),
                "task_losses": task_losses,
                "task_errors": task_errors
            }

            # do_llc_estimation
            if do_llc_estimation:
                x_train = train_features
                y = jax.nn.softmax(model.apply(param, None, x_train))

                rngkey, subkey = jax.random.split(rngkey)
                lambdahat_list = run_llc_estimation(
                        subkey, 
                        loss_fn, 
                        sgld_config, 
                        param, 
                        x_train, 
                        y,
                        itemp=itemp, 
                        loss_trace_minibatch=loss_trace_minibatch,
                        burn_in_prop=burn_in_prop
                    )
                lambdahat = float(np.mean(lambdahat_list))
                
                rec.update(to_json_friendly_tree(
                    {
                        "llc_est": float(lambdahat),
                        "llc_est_list": lambdahat_list,
                        # "loss_trace": loss_trace, 
                    })
                )
            print(
                f"Step {step:6d}, Loss: {loss:.7f}, Test Acc: {test_acc:.7f}"
                + f", llc_est: {lambdahat:.2f}" if do_llc_estimation else ""
            )
            _run.info["records"].append(rec)
        step += 1
        if step >= num_steps:
            break


    ########################################################
    # Task-wise training
    ########################################################

    _run.info["stage_records"] = []
    for stage in range(1, max_num_stages):
        task_ids = np.arange(stage).astype(int)
        stage_mask = (train_features[:, task_ids[0]] == 1)
        
        for task_id in task_ids[1:]:
            stage_mask = stage_mask | (train_features[:, task_id] == 1)
        stage_features = train_features[stage_mask]
        stage_labels = train_labels[stage_mask]
        print(stage_features.shape, stage_labels.shape)

        # reinitialize parameter for each stage
        param_stage = param_init
        optimizer_stage = optax.adam(learning_rate)
        opt_state_stage = optimizer_stage.init(param_stage)

        # Create data generator
        rngkey, gen_key = jax.random.split(rngkey)
        train_gen_stage = data_generator(stage_features, stage_labels, batch_size, gen_key)

        # Training loop
        stage_rec = []
        step_stage = 0
        while step_stage < num_steps:
            x_batch, y_batch = next(train_gen_stage)
            loss, param_stage, opt_state_stage = make_step(param_stage, x_batch, y_batch, opt_state_stage)

            if step_stage % logging_period == 0:
                test_acc = compute_accuracy(param_stage, test_features, test_labels)
                test_loss = loss_fn(param_stage, test_features, test_labels)
                batch_acc = compute_accuracy(param_stage, x_batch, y_batch)
                rec = {
                    "step": step_stage,
                    "loss": float(loss),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                    "batch_acc": float(batch_acc),
                }
                
                stage_rec.append(rec)
                if do_llc_estimation:
                    x_train = stage_features
                    y = jax.nn.softmax(model.apply(param_stage, None, x_train))

                    rngkey, subkey = jax.random.split(rngkey)
                    lambdahat_list = run_llc_estimation(
                        subkey, 
                        loss_fn, 
                        sgld_config, 
                        param_stage, 
                        x_train, 
                        y,
                        itemp=itemp, 
                        loss_trace_minibatch=loss_trace_minibatch,
                        burn_in_prop=burn_in_prop
                    )
                    lambdahat = float(np.mean(lambdahat_list))
                    rec.update(
                        {
                            "llc_est": float(lambdahat),
                            "llc_est_list": lambdahat_list,
                            # "loss_trace": loss_trace, 
                        }
                    )
                print(
                    f"Stage {stage}, "
                    + f"Step {step_stage:6d}, "
                    + f"Loss: {loss:.8f}, "
                    + f"Test Acc: {test_acc:.2f}, "
                    + f"llc_est: {lambdahat:.2f}" if do_llc_estimation else ""
                )
            step_stage += 1
            if step_stage >= num_steps:
                break    
            if step_stage > 0 and stage_rec[-1]["loss"] < early_stopping_epsilon:
                break
        _run.info["stage_records"].append(stage_rec)

    
    return 
    
