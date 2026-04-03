import jax
import jax.numpy as jnp
from jax import grad

# Pauli matrices
Z = jnp.array([[1.0, 0.0],
               [0.0, -1.0]])

X = jnp.array([[0.0, 1.0],
               [1.0, 0.0]])

# Hamiltonian
H = Z + 0.5 * X

# Initial state |0>
psi0 = jnp.array([1.0, 0.0])

# Rotation around Y
def Ry(theta):
    return jnp.array([
        [jnp.cos(theta / 2), -jnp.sin(theta / 2)],
        [jnp.sin(theta / 2),  jnp.cos(theta / 2)]
    ])

# Prepare state
def state(params):
    theta1, theta2 = params
    return Ry(theta2) @ Ry(theta1) @ psi0

# Energy expectation value
def energy(params):
    psi = state(params)
    return jnp.real(jnp.conj(psi) @ (H @ psi))

# Gradient with respect to both parameters
grad_energy = grad(energy)

# Gradient descent
params = jnp.array([0.1, 0.2])
lr = 0.1

for i in range(50):
    g = grad_energy(params)
    params = params - lr * g

    if i % 10 == 0:
        print(f"Step {i}, Energy = {energy(params):.6f}, params = {params}")