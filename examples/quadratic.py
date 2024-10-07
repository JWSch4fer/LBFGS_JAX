import jax
import jax.numpy as jnp
from GradientTransformation import Lbfgs

def n_dim_quadratic(x: jnp.ndarray, dtype= jnp.float32) -> jnp.ndarray:
    """
    general quadratic function with increasing minima

    minimum is increasing integers starting from 0

    """
    coefficients = jnp.arange(x.shape[0])
    result = jnp.sum((-1*coefficients + x)**2)
    return jnp.asarray(result, dtype=dtype)

def main():
    # Define the loss function
    def loss(x):
        return n_dim_quadratic(x)

    # Initialize parameters near the local maximum
    n = 1000
    # Initialize PRNG key
    key = jax.random.PRNGKey(42)

    # Split the key
    key1, _ = jax.random.split(key, 2)    # Initialize PRNG key
    x0 = jax.random.uniform(key1, shape=(n,), minval=-4, maxval=4)

    # Instantiate the L-BFGS optimizer
    optimizer = Lbfgs(f=loss, m=10, max_iter=10000, tol=1e-6)

    # Initialize optimizer state
    opt_state = optimizer.init(x0)
    final_position, loss_history = optimizer.update(opt_state)
    final_value = loss(final_position)

    print("Estimated minimum position:", final_position)
    print("Function value at minimum:", final_value)
    print("_______________________________________")

if __name__ == "__main__":
    main()


