# tests/test_easom.py
import pytest
import jax
import jax.numpy as jnp
from GradientTransformation import Lbfgs
# from optimizer.easom import easom

def test_lbfgs_easom_minimization():
    """
    Test the L-BFGS optimizer on the Easom function to verify it finds the global minimum.
    """
    # Define the loss function
    def easom(x):
        x1, x2 = x
        return -jnp.cos(x1) * jnp.cos(x2) * jnp.exp(-((x1 - jnp.pi)**2 + (x2 - jnp.pi)**2))

    # Initialize parameters near a local maximum to test the optimizer's ability to escape
    x0 = jnp.array([3.0, 3.0])  # Start at (0,0), a local maximum

    # Instantiate the L-BFGS optimizer
    lbfgs = Lbfgs(f=easom, m=10, max_iter=100, tol=1e-6)

    # Initialize optimizer state
    opt_state = lbfgs.init(x0)

    # Run the optimization loop using lax.scan
    x_final, losses = lbfgs.update(opt_state)

    # Extract the final position and function value
    final_position = x_final
    final_value = easom(final_position)

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

def test_lbfgs_rastrigin_minimization():
    """
    Test the L-BFGS optimizer on the Easom function to verify it finds the global minimum.
    """
    # Define the loss function
    def rastrigin(x):
        return 10*x.shape[0] + sum(xi**2 - 10 * jnp.cos(2 * jnp.pi * xi) for xi in x)

    # Initialize parameters near a local maximum to test the optimizer's ability to escape
    x0 = jnp.array([3.0, 3.0])  # Start at (0,0), a local maximum

    # Instantiate the L-BFGS optimizer
    lbfgs = Lbfgs(f=rastrigin, m=10, max_iter=100, tol=1e-6)

    # Initialize optimizer state
    opt_state = lbfgs.init(x0)

    # Run the optimization loop using lax.scan
    x_final, losses = lbfgs.update(opt_state)

    # Extract the final position and function value
    final_position = x_final
    final_value = rastrigin(final_position)

    # Define the expected minimum position and value
    expected_position = jnp.array([0,0])
    expected_value = 0.0

    # Define tolerances
    position_tol = 1e-3
    value_tol = 1e-3

    # Assertions
    assert jnp.allclose(final_position, expected_position, atol=position_tol), \
        f"Final position {final_position} is not close to expected {expected_position}"

    assert jnp.abs(final_value - expected_value) < value_tol, \
        f"Final function value {final_value} is not close to expected {expected_value}"




