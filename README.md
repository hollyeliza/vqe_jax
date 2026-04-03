# vqe_jax

Nikita Astrakhantsev advice at SimQuDyn was to learn how to use JAX. He may be biased but I am taking his advice by exploring JAX through a VQE implementation.

## JAX vs NumPy

NumPy is designed to run on the CPU and executes operations directly, whereas JAX is built to run on CPUs, GPUs, and TPUs. It is very similar to NumPy but can compile computations into efficient low-level code so is better suited for large-scale or performance-critical workloads.

JAX is very similar to NumPy — it uses the same syntax and lets you work with multi-dimensional arrays in a familiar way. NumPy is widely used for numerical computing, but it just executes operations directly on the CPU.

JAX extends this by introducing a different way of thinking about computation. Arrays are immutable and code is written using pure functions. Instead of executing immediately, functions are traced and converted into an intermediate form called a jaxpr.

This allows JAX to automatically differentiate functions, apply transformations, and compile code using just-in-time (JIT) compilation. The result is efficient low-level code that can run on accelerated hardware like GPUs and TPUs.

## Immutable arrays

JAX uses immutable arrays and numpy doesn't. JAX does this because it needs the code to behave like a clean mathematical function rather than a sequence of state changes. If values can be modified in place, it becomes harde to track what depends on what, which breaks things like automatic differentiation and makes optimisation and compilation unreliable.



