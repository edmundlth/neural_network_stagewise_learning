import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional

def compute_parity(subset: jnp.ndarray, bits: jnp.ndarray) -> int:
    """Compute the parity of the specified subset of bits."""
    return jnp.sum(bits[subset]) % 2

def generate_zipfian_probs(n_tasks: int, alpha: float) -> jnp.ndarray:
    """Generate Zipfian distribution probabilities."""
    probs = jnp.power(jnp.arange(1, n_tasks + 1), -(alpha + 1))
    return probs / probs.sum()

def generate_subsets(rng_key: jax.random.PRNGKey, n: int, k: int, n_tasks: int) -> jnp.ndarray:
    """Generate random subsets for each task."""
    keys = random.split(rng_key, n_tasks)
    return jnp.array([random.choice(key, jnp.arange(n), shape=(k,), replace=False) 
                      for key in keys])

def generate_multitask_sparse_parity_dataset(
    rng_key: jax.random.PRNGKey,
    n_tasks: int,
    n: int,
    k: int,
    num_samples: int,
    alpha: float = 0.4, 
    seed: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate a dataset for the multitask sparse parity problem."""
    
    if seed is not None:
        rng_key = random.PRNGKey(seed) # Reset the RNG key to generate the same dataset
    subset_key, data_key = random.split(rng_key)
    subtasks = generate_subsets(subset_key, n, k, n_tasks)
    
    probs = generate_zipfian_probs(n_tasks, alpha)
    
    task_indices = random.choice(data_key, jnp.arange(n_tasks), shape=(num_samples,), p=probs)
    control_bits = jax.nn.one_hot(task_indices, n_tasks)
    task_bits = random.randint(data_key, (num_samples, n), 0, 2)
    
    labels = jax.vmap(compute_parity, in_axes=(0, 0))(
        subtasks[task_indices], 
        task_bits
    )
    
    features = jnp.concatenate([control_bits, task_bits], axis=1)
    
    return features, labels, subtasks

# Run the tests
if __name__ == "__main__":

    def test_multitask_sparse_parity_dataset():
        """Test the multitask sparse parity dataset generation."""
        rng_key = random.PRNGKey(42)
        n_tasks, n, k = 5, 10, 3
        num_samples = 1000
        
        features, labels, subtasks = generate_multitask_sparse_parity_dataset(rng_key, n_tasks, n, k, num_samples)
        
        assert features.shape == (num_samples, n_tasks + n)
        assert labels.shape == (num_samples,)
        assert subtasks.shape == (n_tasks, k)
        
        assert jnp.all(jnp.sum(features[:, :n_tasks], axis=1) == 1)
        assert jnp.all((features[:, n_tasks:] == 0) | (features[:, n_tasks:] == 1))
        
        for i in range(num_samples):
            task_idx = jnp.argmax(features[i, :n_tasks])
            task_bits = features[i, n_tasks:]
            expected_parity = compute_parity(subtasks[task_idx], task_bits)
            assert labels[i] == expected_parity
        
        # Check if subsets are different
        assert not jnp.all(subtasks[0] == subtasks[1:])
        
        print("All tests passed!")

    def test_generate_subsets():
        """Test the generate_subsets function to ensure unique subsets."""
        rng_key = random.PRNGKey(0)
        n, k, n_tasks = 10, 3, 5
        
        subtasks = generate_subsets(rng_key, n, k, n_tasks)
        
        assert subtasks.shape == (n_tasks, k)
        
        # Check for no duplicates within each subset
        for subset in subtasks:
            assert jnp.unique(subset).shape[0] == k
        
        # Check that not all subsets are the same
        assert not jnp.all(subtasks[0] == subtasks[1:])
        
        print("Generate subsets test passed!")

    test_multitask_sparse_parity_dataset()
    test_generate_subsets()