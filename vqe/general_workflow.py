import jax.numpy as jnp
from jax import grad

# Going to use framework to do VQE on different Hamiltonians
# Tilted Holstein

# ---------- Basic matrices ----------
I = jnp.array([[1.0, 0.0],
               [0.0, 1.0]])

X = jnp.array([[0.0, 1.0],
               [1.0, 0.0]])

Y = jnp.array([[0.0, -1.0j],
               [1.0j, 0.0]])

Z = jnp.array([[1.0, 0.0],
               [0.0, -1.0]])


# ---------- Single-qubit gates ----------
def Ry(theta):
    return jnp.array([
        [jnp.cos(theta / 2), -jnp.sin(theta / 2)],
        [jnp.sin(theta / 2),  jnp.cos(theta / 2)]
    ])


def Rz(theta):
    return jnp.array([
        [jnp.exp(-0.5j * theta), 0.0],
        [0.0, jnp.exp(0.5j * theta)]
    ])


# ---------- Problem definition ----------
H = Z + 0.5 * X
psi0 = jnp.array([1.0, 0.0])   # |0>


# ---------- Ansatz ----------
def state(params):
    theta1, theta2 = params
    return Rz(theta2) @ Ry(theta1) @ psi0


# ---------- Objective ----------
def energy(params):
    psi = state(params)
    return jnp.real(jnp.conj(psi) @ (H @ psi))


# ---------- Optimisation ----------
grad_energy = grad(energy)

params = jnp.array([0.1, 0.2])
lr = 0.1

for i in range(70):
    g = grad_energy(params)
    params = params - lr * g

    if i % 10 == 0:
        print(f"Step {i:2d} | Energy = {energy(params):.6f} | params = {params}")