import numpy as np
import jax.numpy as jnp

# np
arr = np.array([[1., 2.],[3., 4.]])
brr = np.array([[5., 6.],[8., 9.]])

# jnp
arr_j = jnp.array([[1., 2.],[3., 4.]])
brr_j = jnp.array([[5., 6.],[8., 9.]])

# print(arr[1])
# arr[1] = [69.0, 2.0] # can do with no error
# print(arr)

# you can mutate data structure directly in numpy
# this is an error in jax as arrays are immutable

print(arr_j)
# arr_j[1] = [69.0, 2.0 # this is error
new_arr_j = arr_j.at[1].set((69.0, 2.0)) # have to make a new array
print(new_arr_j)