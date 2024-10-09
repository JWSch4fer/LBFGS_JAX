# L-BFGS optimizer written with JAX

## Features

- Implements the Limited-memory [BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) algorithm.
- JIT/vmap/pmap compatible for performance with JAX.
- Note requirements.txt is setup for JAX[CPU]

## Installation
```
pip install GradientTransformation
```

## Usage
*NOTE*: This example is a simple quadratic function.\
1000-dimensional Rosenbrock solved in 4038 steps, see repo


Define a function to minimize
```python
from GradientTransformation import Lbfgs

def func(x): 
    jnp.sum((-1*coefficients + x)**2)
```

Call Lbfgs
-f: function to minimize
-m: number of previous iterations to store in memory
-tol: tolerance of convergence
```python
optimizer = Lbfgs(f=func, m=10, tol=1e-6)
```

iterate to find minimum
```python
# Initialize optimizer state
opt_state = optimizer.init(x0)

@jax.jit
def opt_step(carry, _):
    opt_state, losses = carry
    opt_state = optimizer.update(opt_state)
    losses = losses.at[opt_state.k].set(loss(opt_state.position))
    return (opt_state, losses), _

iterations=10000   #<-- A lot of iterations!!!
losses = jnp.zeros((iterations,))
(final_state, losses), _ = jax.lax.scan(opt_step, (opt_state,losses), None, length=iterations)
#note losses will be the length of iterations
losses = jnp.array(jnp.where(losses == 0, jnp.nan, losses))
```

output
```
[-7.577116e-15  1.000000e+00  2.000000e+00  3.000000e+00  4.000000e+00
  5.000000e+00  6.000000e+00  7.000000e+00  8.000000e+00  9.000000e+00
  1.000000e+01  1.100000e+01  1.200000e+01  1.300000e+01  1.400000e+01
  1.500000e+01  1.600000e+01  1.700000e+01  1.800000e+01  1.900000e+01
  2.000000e+01  2.100000e+01  2.200000e+01  2.300000e+01  2.400000e+01
  2.500000e+01  2.600000e+01  2.700000e+01  2.800000e+01  2.900000e+01
  3.000000e+01  3.100000e+01  3.200000e+01  3.300000e+01  3.400000e+01
  3.500000e+01  3.600000e+01  3.700000e+01  3.800000e+01  3.900000e+01
  4.000000e+01  4.100000e+01  4.200000e+01  4.300000e+01  4.400000e+01
  4.500000e+01  4.600000e+01  4.700000e+01  4.800000e+01  4.900000e+01
  5.000000e+01  5.100000e+01  5.200000e+01  5.300000e+01  5.400000e+01
  5.500000e+01  5.600000e+01  5.700000e+01  5.800000e+01  5.900000e+01
  6.000000e+01  6.100000e+01  6.200000e+01  6.300000e+01  6.400000e+01
  6.500000e+01  6.600000e+01  6.700000e+01  6.800000e+01  6.900000e+01
  7.000000e+01  7.100000e+01  7.200000e+01  7.300000e+01  7.400000e+01
  7.500000e+01  7.600000e+01  7.700000e+01  7.800000e+01  7.900000e+01
  8.000000e+01  8.100000e+01  8.200000e+01  8.300000e+01  8.400000e+01
  8.500000e+01  8.600000e+01  8.700000e+01  8.800000e+01  8.900000e+01
  9.000000e+01  9.100000e+01  9.200000e+01  9.300000e+01  9.400000e+01
  9.500000e+01  9.600000e+01  9.700000e+01  9.800000e+01  9.900000e+01]

Function value at minimum: 5.7412694e-29
k:  2   #<-- stops early if gradient norm is less than tol!!
```


