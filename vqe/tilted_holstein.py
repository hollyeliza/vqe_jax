# Okay so I can fist convert into Pauli strings
# Look into a good way to do this 

import jax
import jax.numpy as jnp
from jax import grad

# Pauli matrices
Z = jnp.array([[1.0, 0.0],
               [0.0, -1.0]])

X = jnp.array([[0.0, 1.0],
               [1.0, 0.0]])