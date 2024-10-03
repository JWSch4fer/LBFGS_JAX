# tests/test_easom.py
import pytest
import jax
import jax.numpy as jnp
from optimizer.lbfgs import LbfgsGradientTransformation
from optimizer.easom import easom

def test_lbfgs_easom_minimization():
    """
    Test the L-BFGS optimizer on the Easom function to verify it finds the global minimum.
    """
    # Define the loss function
    def loss(x):
        return easom(x)

    # Initialize parameters near a local maximum to test the optimizer's ability to escape
    x0 = jnp.array([0.0, 0.0])  # Start at (0,0), a local maximum

    # Instantiate the L-BFGS optimizer
    lbfgs_opt = LbfgsGradientTransformation(f=loss, m=10, max_iter=100, tol=1e-6)

    # Initialize optimizer state
    opt_state = lbfgs_opt.init(x0)

    # Define the optimization step
    @jax.jit
    def opt_step(iterate):
        x, opt_state = iterate
        grad = jax.grad(loss)(x)
        updates, new_opt_state = lbfgs_opt.update(grad, opt_state, x)
        x = x + updates  # L-BFGS computes the actual update
        return (x, new_opt_state), loss(x)

    # Define the number of iterations
    iterations = 1000

    # Run the optimization loop using lax.scan
    (x_final, final_opt_state), losses = jax.lax.scan(opt_step, (x0, opt_state), None, length=iterations)

    # Extract the final position and function value
    final_position = x_final
    final_value = loss(final_position)

    # Define the expected minimum position and value
    expected_position = jnp.array([jnp.pi, jnp.pi])
    expected_value = -1.0

    # Define tolerances
    position_tol = 1e-3
    value_tol = 1e-3

    # Assertions
    assert jnp.allclose(final_position, expected_position, atol=position_tol), \
        f"Final position {final_position} is not close to expected {expected_position}"

    assert jnp.abs(final_value - expected_value) < value_tol, \
        f"Final function value {final_value} is not close to expected {expected_value}"

