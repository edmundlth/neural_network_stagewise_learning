import jax
import jax.numpy as jnp
import jax.tree_util as jtree

import numpy as np
import optax
import ast

from dln import (
    create_dln_model, 
    create_minibatches, 
    true_dln_learning_coefficient, 
    mse_loss, 
    get_dln_total_product_matrix
)

from sgld_utils import (
    SGLDConfig, 
    run_sgld, 
    run_sgld_known_potential
)
from utils import to_json_friendly_tree, running_mean
import os
import json
import functools

from sacred import Experiment
# Create a new experiment
ex = Experiment('dln_stagewise_learning')





def init_teacher_matrix(
        input_dim, 
        output_dim, 
        config={"type":"diagonal", "config_vals":[50, 10]}
    ):
    teacher_matrix_type = config["type"]
    config_vals = config["config_vals"]
    if isinstance(config_vals, str):
        config_vals = ast.literal_eval(config_vals)
    print(f"Initialising teacher matrix with config: {config}")
    num_modes = min(input_dim, output_dim)
    if teacher_matrix_type == "random":
        teacher_matrix = np.random.randn(output_dim, input_dim)
    elif teacher_matrix_type == "diagonal":
        max_val, min_val = config_vals
        teacher_matrix = np.diag(np.linspace(max_val, min_val, num_modes))
    elif teacher_matrix_type == "diag_power_law":
        power, max_val = config_vals
        spectra = [max_val * (i + 1) ** (-power) for i in range(num_modes)]
        teacher_matrix = np.diag(spectra)
    elif teacher_matrix_type == "band":
        bandwidth, bandgap = config_vals[:2]
        if len(config_vals) == 3:
            num_group = config_vals[2]
        else:
            num_group = 3
        spectra = np.zeros(num_modes)
        current_band_min = 0.1
        num_assigned_modes = 0
        while num_assigned_modes < num_modes:
            num_modes_in_band = np.random.randint(2, num_modes // min(num_group, num_modes))
            num_modes_in_band = min(num_modes_in_band, num_modes - num_assigned_modes)
            current_band_max = current_band_min + bandwidth
            singular_values = np.random.uniform(current_band_min, current_band_max, num_modes_in_band)
            spectra[num_assigned_modes:num_assigned_modes + num_modes_in_band] = singular_values
            num_assigned_modes += num_modes_in_band
            current_band_min = current_band_max + bandgap + np.random.uniform(bandgap / 10, bandgap / 5)
        teacher_matrix = np.diag(spectra)
    else:
        raise ValueError(f"Unknown teacher matrix type: {teacher_matrix_type}")
    print(f"Teacher matrix shape: {teacher_matrix.shape}")
    print(f"Teacher matrix spectrum: {np.linalg.eigvals(teacher_matrix)}")
    return teacher_matrix[:input_dim, :output_dim]


def estimate_cross_correlation_matrix(X, Y):
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
    n = X.shape[0]
    # Compute the non-centered cross covariance matrix
    C_XY = jnp.dot(X.T, Y) / n
    return C_XY




def generate_random_covariance_matrix(key, n, max_variance, min_variance=1.0):
    """
    Generate a random covariance matrix with controlled variances using JAX.
    
    Parameters:
    key (PRNGKey): JAX random number generator key
    n (int): The dimension of the covariance matrix
    max_variance (float): The maximum variance (must be greater than min_variance)
    min_variance (float): The minimum variance (default is 1.0)
    
    Returns:
    jax.numpy.ndarray: A random covariance matrix
    """
    # if max_variance <= min_variance:
    #     raise ValueError("max_variance must be greater than min_variance")

    # Split the random key
    key_eigvals, key_q = jax.random.split(key)

    # Generate random eigenvalues (variances)
    log_eigvals = jax.random.uniform(key_eigvals, 
                                 shape=(n,), 
                                 minval=jnp.log(min_variance), 
                                 maxval=jnp.log(max_variance))
    eigvals = jnp.exp(log_eigvals)
    
    # Generate a random orthogonal matrix using QR decomposition
    q_matrix, _ = jnp.linalg.qr(jax.random.normal(key_q, shape=(n, n)))
    
    # Construct the covariance matrix
    C = q_matrix @ jnp.diag(eigvals) @ q_matrix.T
    
    # Ensure symmetry (can be slightly off due to numerical precision)
    C = (C + C.T) / 2
    
    return C

def generate_correlated_data(key, n_samples, correlation_matrix):
    """
    Generate data with a specified correlation structure using JAX multivariate normal.
    
    :param key: JAX random key
    :param n_samples: Number of samples to generate
    :param correlation_matrix: The desired correlation matrix
    :return: Generated data with the specified correlation structure
    """
    n_features = correlation_matrix.shape[0]
    mean = jnp.zeros(n_features)
    return jax.random.multivariate_normal(key, mean, correlation_matrix, (n_samples,))


def make_potential_fn(
        teacher_matrix, 
        feature_corr, 
        feature_output_cross_correlation, 
    ):
    eigvals, eigvecs = jnp.linalg.eigh(feature_corr)
    ChangeOfBasis = eigvecs @ jnp.diag(eigvals ** (-1/2)) @ eigvecs.T
    modified_feature_output_cross_correlation = feature_output_cross_correlation @ ChangeOfBasis 
    U, S, V = jnp.linalg.svd(modified_feature_output_cross_correlation)
    V = V.T
    Vhat = jnp.linalg.inv(ChangeOfBasis) @ V
    def get_matrices():
        return U, S, V, Vhat, ChangeOfBasis

    @jax.jit
    def potential_matrix_fn(param):
        param = jtree.tree_map(lambda x: jnp.array(x), param, is_leaf=is_leaf)
        total_matrix = get_dln_total_product_matrix(param)
        potential_matrix = U.T @ (total_matrix.T - teacher_matrix) @ Vhat
        return potential_matrix
    
    @jax.jit
    def potential_fn(param):
        potential_matrix = potential_matrix_fn(param)
        return jnp.sum(potential_matrix ** 2)
    return potential_matrix_fn, potential_fn, get_matrices


def is_leaf(x):
    return isinstance(x, (jnp.ndarray, list, np.ndarray))

    

def gradient_flow(
        potential_fn, 
        initial_params, 
        num_steps, 
        learning_rate=1e-3, 
        logging_period=100, 
        early_stopping_epsilon=None,
        min_num_steps=None, 
        eval_fns = None,
    ):
    """
    Perform gradient flow on the given potential function.
    
    :param potential_fn: The potential function to optimize (should take params as input and return a scalar)
    :param initial_params: Initial parameters (can be a pytree)
    :param num_steps: Number of optimization steps
    :param learning_rate: Learning rate for the gradient descent (default: 1e-3)
    :return: Tuple of (optimized parameters, loss history)
    """
    if eval_fns is None:
        eval_fns = [potential_fn]
    
    # Initialize the optimizer and the gradient function
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(initial_params)
    grad_fn = jax.grad(potential_fn)
    
    # Define a single step of the optimization
    @jax.jit
    def step(params, opt_state):
        loss = potential_fn(params)
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Initialize parameters and loss history
    params = initial_params
    records = []

    # Run the optimization loop
    for t in range(num_steps):
        params, opt_state, loss = step(params, opt_state)
        if t % logging_period == 0:
            rec = {
                "t": t,
                "loss": float(loss),
                "evals": [eval_fn(params) for eval_fn in eval_fns]
            }
            records.append(to_json_friendly_tree(rec))
            
            if not jnp.isfinite(loss):
                print("Loss is not finite. Stopping the optimization.")
                break

            if early_stopping_epsilon is not None and loss < early_stopping_epsilon:
                if min_num_steps is None or t >= min_num_steps:
                    print(f"LOSS BELOW EARLY STOPPING EPSILON. STOPPING EARLY. t={t}")
                    break
                
    if t % logging_period != 0: # Log the final step
        rec = {
            "t": t,
            "loss": float(loss), 
            "evals": [eval_fn(params) for eval_fn in eval_fns]
        }
        records.append(to_json_friendly_tree(rec))
    return records, params


def run_sgld_chain(
        rngkey, 
        loss_fn, 
        sgld_config, 
        param_init, 
        x, 
        y,
        itemp=None, 
        trace_batch_loss=False
    ):
    """
    Run SGLD with a known potential function.
    :return: Tuple of (loss_trace, distances
    """
    rngkey, subkey = jax.random.split(rngkey)
    rngkey, subkey = jax.random.split(rngkey)
    loss_trace, distances, acceptance_probs = run_sgld(
        subkey, 
        loss_fn, 
        sgld_config, 
        param_init, 
        x, 
        y,
        itemp=itemp, 
        trace_batch_loss=trace_batch_loss, 
        compute_distance=False, 
        verbose=False, 
        compute_mala_acceptance=False, 
        output_samples=False
    )
    return loss_trace

def run_llc_estimation(
        rngkey, 
        loss_fn, 
        sgld_config, 
        param_init, 
        x, 
        y,
        itemp=1.0, 
        loss_trace_minibatch=True,
        burn_in_prop=0.9,
    ):
    num_training_data = x.shape[0]
    lambdahat_list = []
    for chain_idx, chain_rngkey in enumerate(jax.random.split(rngkey, sgld_config.num_chains)):
        loss_trace = run_sgld_chain(
            chain_rngkey, 
            loss_fn, 
            sgld_config, 
            param_init, 
            x, 
            y,
            itemp=itemp, 
            trace_batch_loss=loss_trace_minibatch
        )
    
        trace_start = min(int(burn_in_prop * len(loss_trace)), len(loss_trace) - 1)
        init_loss = loss_fn(param_init, x, y)
        lambdahat = float(np.mean(loss_trace[trace_start:]) - init_loss) * num_training_data * itemp
        lambdahat_list.append(lambdahat)
    lambdahat = np.mean(lambdahat_list)
    return lambdahat_list


@ex.config
def cfg():
    expt_name = f"dev"
    use_behavioural = True
    do_llc_estimation = False
    spectral_resolution = None
    sgld_config = {
        'epsilon': 2e-8,
        'gamma': 1.0,
        'num_steps': 500,
        "num_chains": 1, 
        "batch_size": 512
    }
    loss_trace_minibatch = True
    burn_in_prop = 0.9
    data_config = {
        "num_training_data": 10000,
        "feature_map": None, # None, ("polynomial", d)
        "output_noise_std": 0.1, 
        "input_variance_range": (1.0, 10.0), 
        # ("diagonal", max, min), 
        # ("diag_power_law", power, max), 
        # ("band", bandwidth, bandgap, num_group)
        "teacher_matrix": {
            "type": "diagonal",
            "config_vals": (50, 10),
        }, 
        "idcorr": True,
    }
    model_config = {
        "input_dim": 3,
        "output_dim": 3,
        "hidden_layer_widths": [3, 3],
        "initialisation_exponent": None,
        "init_origin": True,
    }

    itemp = None
    training_config = {
        "optim": "sgd", 
        "learning_rate": 5e-5, 
        "momentum": 0.9,
        "batch_size": 128, 
        "num_steps": 20000, 
        "min_num_steps": 2000,
        "early_stopping_loss_threshold": 0.001,
    }
    gd_training_config = {
        "learning_rate": training_config["learning_rate"] / 4,
        "early_stopping_epsilon": 5e-4,
        "min_num_steps": training_config["min_num_steps"],
        "num_steps": training_config["num_steps"],
    }
    seed = 42
    logging_period = 200
    log_full_checkpoint_param = False
    eval_mode = "minimal" # "full", "minimal"
    do_plotting = False
    verbose = True




@ex.automain
def run_experiment(
    _run, 
    expt_name,
    use_behavioural,
    do_llc_estimation,
    spectral_resolution,
    sgld_config,
    loss_trace_minibatch,
    burn_in_prop,
    data_config,
    model_config,
    itemp,
    training_config,
    gd_training_config,
    seed,
    logging_period,
    log_full_checkpoint_param,
    eval_mode,
    do_plotting, 
    verbose,
):
    # seeding
    np.random.seed(seed)
    rngkey = jax.random.PRNGKey(seed)


    ####################
    # Parse configs
    ####################
    num_training_data = data_config["num_training_data"]
    input_variance_range = data_config["input_variance_range"]
    output_noise_std = data_config["output_noise_std"]
    use_idcorr = data_config["idcorr"]
    input_dim = model_config["input_dim"]
    output_dim = model_config["output_dim"]
    initorigin = model_config["init_origin"]
    hidden_layer_widths = model_config["hidden_layer_widths"]
    if isinstance(hidden_layer_widths, str):
        hidden_layer_widths = ast.literal_eval(hidden_layer_widths)
    initialisation_exponent = model_config["initialisation_exponent"]
    if initialisation_exponent is None:
        if initorigin:
            initialisation_exponent = 3.0
        else: 
            initialisation_exponent = -2.0
    if itemp is None:
        itemp = 1 / np.log(num_training_data)


    num_modes = min(input_dim, output_dim)
    num_hidden_layers = len(hidden_layer_widths)
    layer_widths = hidden_layer_widths + [output_dim]
    average_width = np.mean(layer_widths)



    ####################
    # Initialisations
    ####################
    # Teacher matrix
    teacher_matrix = init_teacher_matrix(input_dim, output_dim, config=data_config["teacher_matrix"])
    
    # Gerenate training data
    rngkey, rngkey = jax.random.split(rngkey)
    if use_idcorr:
        input_correlation_matrix = jnp.eye(input_dim)
    else:
        input_correlation_matrix = generate_random_covariance_matrix(
            rngkey, 
            input_dim, 
            max_variance=input_variance_range[1],
            min_variance=input_variance_range[0]
        )
    x_train = jax.random.multivariate_normal(
        rngkey, 
        jnp.zeros(input_dim), 
        input_correlation_matrix, 
        shape=(num_training_data,), 
        dtype=jnp.float32
    )

    rngkey, rngkey = jax.random.split(rngkey)
    y_train = (
        x_train @ teacher_matrix 
        + jax.random.normal(rngkey, shape=(num_training_data, output_dim)) * output_noise_std
    )

    # Create DLN model
    initialisation_sigma = np.sqrt(average_width ** (-initialisation_exponent))
    model = create_dln_model(layer_widths, sigma=initialisation_sigma)
    loss_fn = jax.jit(lambda param, inputs, targets: mse_loss(param, model, inputs, targets))

    rngkey, subkey = jax.random.split(rngkey)
    param_init = model.init(rngkey, jnp.zeros((1, input_dim)))

    print("Model initialised with shapes:")
    print(json.dumps(jtree.tree_map(lambda x: x.shape, param_init), indent=2))


    ##############################################
    # Useful matrices and functions
    ##############################################
    input_output_cross_correlation_matrix = teacher_matrix @ input_correlation_matrix
    potential_matrix_fn, potential_fn, get_matrices = make_potential_fn(
        teacher_matrix, 
        input_correlation_matrix, 
        input_output_cross_correlation_matrix
    )
    # est_input_output_correlation_matrix = (x_train.T @ y_train) / num_training_data
    # est_input_correlation_matrix = (x_train.T @ x_train) / num_training_data
        
    U, S, V, Vhat, ChangeOfBasis = get_matrices()
    if spectral_resolution is not None:
        previous_sing_val = S[0]
        indices = [0]
        for i, sing_val in enumerate(S[1:], 1):
            if np.abs(sing_val - previous_sing_val) > spectral_resolution:
                indices.append(i)
                previous_sing_val = sing_val

    POTENTIAL_TYPES = [ 
        ("block", num_modes), 
        ("diag", num_modes), 
        ("row_col", num_modes),
        # ("behavioural", num_modes),
    ]
    if eval_mode == "full":
        POTENTIAL_TYPES += [
            ("col", num_modes), 
            ("row", num_modes),
            ("corner", num_modes),
            ("offdiag_inclusive", num_modes + 1), 
            ("offdiag_exclusive", num_modes + 1), 
        ]
    def get_stage_potential_fn(alpha, potential_type="block"):
        potential_type = potential_type.lower()
        if potential_type == "block":
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                return jnp.sum(H[:alpha + 1, :alpha + 1])
        elif potential_type in ["diag", "diagonal"]:
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                return jnp.sum(jnp.diag(H)[:alpha + 1])
        elif potential_type in ["col", "column", "columns", "cols"]:
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                return jnp.sum(H[:, :alpha + 1])
        elif potential_type in ["row", "rows"]:
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                return jnp.sum(H[:alpha + 1, :])
        elif potential_type in ["corner"]:
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                result = H[alpha, alpha] + H[alpha, :alpha].sum() + H[:alpha, alpha].sum()
                return result
        elif potential_type in ["offdiag_inclusive"]:
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                result = H.sum() - jnp.diag(H)[alpha:].sum()
                return result
        elif potential_type in ["offdiag_exclusive"]:
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                result = jnp.where(
                    alpha > 0,
                    jnp.diag(H)[:alpha].sum(), 
                    H.sum() - jnp.diag(H).sum() # offdiagonal sum
                )
                return result
        elif potential_type in ["row_col"]:
            def stage_potential_fn(param):
                H = potential_matrix_fn(param)**2
                result = H.sum() - H[alpha + 1:, alpha + 1:].sum()
                return result
        elif potential_type in ["behavioural"]:
            def stage_potential_fn(param):
                modified_S = jnp.diag(
                    jnp.array([s if i < alpha else 0 for i, s in enumerate(S)])
                )
                H = (potential_matrix_fn(param) + modified_S) ** 2
                result = jnp.sum(H)
                return result
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
        return jax.jit(stage_potential_fn)
    
    @jax.jit
    def theoretical_potential_gradient_component(param, a, b):
        func = lambda param_other: (potential_matrix_fn(param_other)**2)[a, b]
        g = jax.grad(func)(param)
        return jnp.hstack([entry.flatten() for entry in jtree.tree_flatten(g)[0]])

    @jax.jit
    def theoretical_total_potential_grad_norm(param):
        func = lambda param_other: jnp.sum((potential_matrix_fn(param_other)**2))
        grad = jax.grad(func)(param)
        norm_sq = jtree.tree_reduce(lambda x, y: x + jnp.sum(y**2), grad, 0)
        return jnp.sqrt(norm_sq)


    def theoretical_gradient_norm(param, entries):
        grad = theoretical_potential_gradient_component(param, *entries[0])
        for entry in entries[1:]:
            grad = grad + theoretical_potential_gradient_component(param, *entry)
        norm_sq = jtree.tree_reduce(lambda x, y: x + jnp.sum(y**2), grad, 0)
        return jnp.sqrt(norm_sq)


    ##############################################
    # Record stuff before training
    ##############################################
    _run.info = {
        "expt_properties": {
            "teacher_matrix": teacher_matrix.tolist(),
            "input_correlation_matrix": input_correlation_matrix.tolist(),
            "input_output_cross_correlation_matrix": input_output_cross_correlation_matrix.tolist(),
            "itemp": float(itemp),
            "svd_matrices": {
                "U": U.tolist(),
                "S": S.tolist(),
                "V": V.tolist(),
                "Vhat": Vhat.tolist(),
                "ChangeOfBasis": ChangeOfBasis.tolist()
            },
            "num_hidden_layers": num_hidden_layers,
            "stage_potential_types": list(POTENTIAL_TYPES),
            # "est_input_output_correlation_matrix": est_input_output_correlation_matrix.tolist(),
            # "est_input_correlation_matrix": est_input_correlation_matrix.tolist(),
        }
    }

    ##############################################
    # SGD training
    ##############################################
    sgld_config = SGLDConfig(**sgld_config)
    if training_config["optim"] == "sgd":
        optimizer = optax.sgd(learning_rate=training_config["learning_rate"])
    elif training_config["optim"] == "momentum":
        optimizer = optax.sgd(learning_rate=training_config["learning_rate"], momentum=training_config["momentum"])
    elif training_config["optim"] == "adam":
        optimizer = optax.adam(learning_rate=training_config["learning_rate"], b1=0.9, b2=0.999, eps=1e-8)
    else:
        raise ValueError(f"Unknown optimiser: {training_config['optim']}")
    
    
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))
    trained_param = param_init
    opt_state = optimizer.init(trained_param)

    # Create a list of eval functions
    eval_list = [ # (name, function)
        (
            "full_potential", 
            potential_fn
        ),
        (
            "total_potential_grad_norm", 
            theoretical_total_potential_grad_norm
        ),
        (
            "corrected_total_matrix_diagonals", 
            lambda x: jnp.diag(U.T @ get_dln_total_product_matrix(x).T @ Vhat)
        )
    ]
    for stage_potential_type, num_stages in POTENTIAL_TYPES:
        name = f"stage_potential={stage_potential_type}"
        # We create this list so that the functions are jitted only once
        # We pass it as a default argument to avoid late binding closure issues
        fn_list = [
            get_stage_potential_fn(a, potential_type=stage_potential_type) 
            for a in range(num_stages)
        ]
        func = lambda x, fn_list=fn_list:[
            stage_potential_fn(x)
            for stage_potential_fn in fn_list
        ]
        eval_list.append((name, func))
    if eval_mode == "full":
        eval_list += [
            (
                "potential_matrix", 
                potential_matrix_fn
            ),
            (
                "component_potential_grad_norm", 
                lambda x: [
                    [
                        float(theoretical_gradient_norm(x, [[a, b]]))
                        for b in range(num_modes)
                    ] 
                    for a in range(num_modes)
                ]
            ), 
        ]

    #TODO: record grad norms of other submatrices of the potential matrix.
    print(f"Eval list: {[eval_name for eval_name, _ in eval_list]}")

    # training loop
    t = 0
    _run.info["sgd_logs"] = {
        "eval_list": [eval_name for eval_name, _ in eval_list],
    }
    _run.info["sgd_logs"]["checkpoint_logs"] = []
    max_steps = training_config["num_steps"]
    early_stopping_reached = False
    delay_early_stop = 0
    nan_detected = False
    while t < max_steps:
        for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=training_config["batch_size"]):
            train_loss, grads = grad_fn(trained_param, x_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            trained_param = optax.apply_updates(trained_param, updates)
            
            # logging checkpoint
            if t % logging_period == 0:
                # log total matrix
                total_matrix = get_dln_total_product_matrix(trained_param)
                corrected_total_matrix = U.T @ total_matrix.T @ Vhat
                
                # Estimate rank and using it to calculate true lambda and multiplicity
                try: 
                    est_total_rank = jnp.sum(jnp.abs(jnp.linalg.eigvals(total_matrix)) > min(S) * 1e-2)
                    true_lambda, true_multiplicity = true_dln_learning_coefficient(
                        est_total_rank, 
                        layer_widths, 
                        input_dim, 
                    )
                except Exception as e: # Likely a NotImplementedError from jax.numpy.linalg.eigvals
                    print(f"Error in calculating true lambda and multiplicity: {e}")
                    est_total_rank = None
                    true_lambda, true_multiplicity = None, None

                rec = {
                    "t": t + 1, 
                    "train_loss": float(train_loss),
                    "true_lambda": true_lambda, 
                    "true_multiplicity": true_multiplicity, 
                    "est_total_rank": est_total_rank,
                    "evals": [eval_fn(trained_param) for _, eval_fn in eval_list]
                }
                if eval_mode == "full":
                    rec["total_matrix"] = total_matrix
                    rec["corrected_total_matrix"] = corrected_total_matrix
                    

                if log_full_checkpoint_param:
                    rec["trained_param"] = trained_param
                
                if do_llc_estimation:
                    y_realisable = model.apply(trained_param, x_train) #+ jax.random.normal(subkey, shape=(num_training_data, output_dim)) * output_noise_std
                    if use_behavioural:
                        y = y_realisable
                    else: 
                        y = y_train

                    rngkey, subkey = jax.random.split(rngkey)
                    lambdahat_list = run_llc_estimation(
                        subkey, 
                        loss_fn, 
                        sgld_config, 
                        trained_param, 
                        x_train, 
                        y,
                        itemp=itemp, 
                        loss_trace_minibatch=loss_trace_minibatch,
                        burn_in_prop=burn_in_prop
                    )
                    lambdahat = float(np.mean(lambdahat_list))

                    rec.update(
                        {
                            "lambdahat": lambdahat,
                            "lambdahat_std": float(np.std(lambdahat_list)),
                            "lambdahat_list": lambdahat_list
                        }
                    )
                    if eval_mode == "full":
                        rec["loss_trace"] = loss_trace
                
                if verbose:
                    print(
                        f"t: {t + 1:6d}, "
                        + f"train_loss: {float(train_loss):.3f}, "
                        + (f"llc: {lambdahat:.3f}, " if do_llc_estimation else "")
                        + (f"Est total rank: {est_total_rank}")
                    )

                _run.info["sgd_logs"]["checkpoint_logs"].append(to_json_friendly_tree(rec))
            
            t += 1
            if t >= max_steps:
                print(f"Reached max steps. Stopping. t={t}")
                break

            if train_loss < training_config["early_stopping_loss_threshold"] and t >= training_config["min_num_steps"]:
                if delay_early_stop == 0:
                    print(f"Loss {float(train_loss)} below threshold. Stopping early. t={t}")
                    early_stopping_reached = True
                delay_early_stop += 1
        if early_stopping_reached and delay_early_stop > logging_period * 30:
            # Stop early if the loss is below the threshold for 30 logging periods
            break

        if jnp.isnan(train_loss):
            print("Loss is NaN. Stopping the optimization.")
            nan_detected = True
            break
    _run.info["expt_properties"]["early_stopping_reached"] = early_stopping_reached
    _run.info["expt_properties"]["nan_detected"] = nan_detected


    ##############################################
    # Gradient Descent
    ##############################################
    print("Starting gradient descent")
    gd_early_stopping_epsilon = gd_training_config["early_stopping_epsilon"]
    gd_max_num_steps = gd_training_config["num_steps"]
    gd_min_num_steps = min(gd_training_config["min_num_steps"], gd_max_num_steps)
    gd_learning_rate = gd_training_config["learning_rate"]
    grad_flow_rec, _ = gradient_flow(
        potential_fn, 
        param_init, 
        gd_max_num_steps, 
        learning_rate=gd_learning_rate, 
        logging_period=logging_period, 
        early_stopping_epsilon=gd_early_stopping_epsilon, 
        min_num_steps=gd_min_num_steps,
        eval_fns=[fn for _, fn in eval_list]
    )

    _run.info["gd_logs"] = {
        "checkpoint_logs": grad_flow_rec,
        "eval_list": [eval_name for eval_name, _ in eval_list],
        "max_num_steps": gd_max_num_steps,
        "min_num_steps": gd_min_num_steps,
        "early_stopping_epsilon": gd_early_stopping_epsilon,
        "learning_rate": gd_learning_rate,
    }
    print("Gradient descent completed.")

    ##############################################
    # Stage-wise Gradient Descent
    ##############################################
    print("Starting stagewise gradient descent")
    _run.info["stagewise_gd_logs"] = {}
    for stage_potential_type, num_stages in POTENTIAL_TYPES:
        stage_potential_fn_list = [
            get_stage_potential_fn(alpha, potential_type=stage_potential_type) 
            for alpha in range(num_stages)
        ]
        eval_fns = [potential_fn] + stage_potential_fn_list
        eval_names = ["total_potential"] + [
            f"stage_potential_{stage_potential_type}_{alpha}" 
            for alpha in range(num_stages)
        ]
        param = param_init
        _run.info["stagewise_gd_logs"][stage_potential_type] = {
            "eval_lists": list(eval_names),
            "num_stages": len(stage_potential_fn_list), 
            "stage_logs": []
        }
        for alpha, stage_potential_fn in enumerate(stage_potential_fn_list):
            grad_flow_rec, param = gradient_flow(
                stage_potential_fn, 
                param, 
                gd_max_num_steps, 
                learning_rate=gd_learning_rate, 
                logging_period=logging_period, 
                early_stopping_epsilon=gd_early_stopping_epsilon, 
                min_num_steps=gd_min_num_steps,
                eval_fns=eval_fns
            )
            if verbose:
                print(f"Stage {alpha + 1} for potential type `{stage_potential_type}` completed.")
            _run.info["stagewise_gd_logs"][stage_potential_type]["stage_logs"].append(
                to_json_friendly_tree(grad_flow_rec)
            )

            if do_llc_estimation:
                if verbose:
                    print(f"Estimating llc for stage {alpha + 1} for potential type `{stage_potential_type}`")
                y_realisable = model.apply(param, x_train)
                if use_behavioural:
                    y = y_realisable
                else: 
                    y = y_train

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
                lambdahat = np.mean(lambdahat_list)

                
                rngkey, subkey = jax.random.split(rngkey)
                lambdahat_stage_potential_list = []
                for chain_idx, chain_rngkey in enumerate(jax.random.split(subkey, sgld_config.num_chains)):

                    loss_trace, _, _ = run_sgld_known_potential(
                        chain_rngkey, 
                        stage_potential_fn, 
                        sgld_config,
                        param,
                        num_training_data=num_training_data,
                        itemp=itemp,
                        compute_distance=False,
                        verbose=False,
                    )
                    trace_start = min(int(burn_in_prop * len(loss_trace)), len(loss_trace) - 1)
                    lambdahat_stage_potential = float(
                        np.mean(loss_trace[trace_start:]) - loss_trace[0]
                    ) * num_training_data * itemp
                    lambdahat_stage_potential_list.append(lambdahat_stage_potential)
                lambdahat_stage_potential = np.mean(lambdahat_stage_potential_list)

                _run.info["stagewise_gd_logs"][stage_potential_type].update(
                    {
                        f"stage_llc_{alpha}": float(lambdahat),
                        f"stage_llc_list_{alpha}": lambdahat_list,
                        f"stage_llc_known_potential_{alpha}": float(lambdahat_stage_potential),
                        f"stage_llc_known_potential_list_{alpha}": lambdahat_stage_potential_list,
                    }
                )
    return 


