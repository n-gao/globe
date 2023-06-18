import functools
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def tree_scale(scalar, tree):
    return jtu.tree_map(lambda x: scalar * x, tree)

tree_mul = functools.partial(jtu.tree_map, jax.lax.mul)
tree_add = functools.partial(jtu.tree_map, jnp.add)
tree_sub = functools.partial(jtu.tree_map, jnp.subtract)
tree_leave_sum = functools.partial(jtu.tree_map, jnp.sum)
tree_sum = functools.partial(jtu.tree_reduce, jnp.add)


def tree_dot(a, b):
    return tree_sum(tree_leave_sum(tree_mul(a, b)))
