import jax
import jax.numpy as jnp
import numpy as np

from globe.utils import adj_idx


def pair_mask(spins: np.ndarray, diag: bool, drop_diagonal: bool = False, drop_off_block: bool = False) -> np.ndarray:
    """
    Compute a index mask for segment sums over pairwise terms.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    diag: A boolean indicating whether to select from the diagonal elements or offdiagonal.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    A 1D array representing the batched pair mask.
    """
    i, j, _ = adj_idx(jnp.sum(spins, -1))
    result = sort_by_same_spin(i, spins, drop_diagonal, drop_off_block)
    n_same = n_pair_same(spins, drop_diagonal, drop_off_block)
    if diag:
        return result[:n_same]
    else:
        return result[n_same:]


def pair_graph_mask(spins: np.ndarray, diag: bool, drop_diagonal: bool = False, drop_off_block: bool = False) -> np.ndarray:
    """
    Computes a index mask indicating for the pairwise terms to which graph they belong.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    diag: A boolean indicating whether to include diagonal elements in the mask.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    A 1D array representing the pair graph mask.
    """
    _, _, m = adj_idx(np.sum(spins, -1))
    result = sort_by_same_spin(m, spins, drop_diagonal, drop_off_block)
    n_same = n_pair_same(spins, drop_diagonal, drop_off_block)
    if diag:
        return result[:n_same]
    else:
        return result[n_same:]


def pair_block_mask(spins: np.ndarray, drop_diagonal: bool = False, drop_off_block: bool = False) -> np.ndarray:
    """
    Computes a index mask that indicates to which block a pairwise term belongs.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    A 1D array representing the pair block mask.
    """
    result = []
    for a, b in spins:
        mask = np.block([
            [np.ones((a, a)), np.zeros((a, b))],
            [np.zeros((b, a)), np.ones((b, b))]
        ])
        if drop_diagonal:
            # set diagonal to -1
            mask -= 2 * np.eye(a+b)
        if drop_off_block:
            mask[np.tril_indices(a+b)] = -1
        mask = mask.reshape(-1)
        # Remove potential diagonal elements; also reshapes to 1D array
        result.append(mask[mask >= 0].astype(bool))
    return np.concatenate(result)


def sort_by_same_spin(pairs: jax.Array, spins: np.ndarray, drop_diagonal: bool = False, drop_off_block: bool = False) -> jax.Array:
    """
    Rearranges pairwise terms such that the block diagonals are first.

    Args:
    pairs: A 1D array representing the pairwise terms.
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    A 1D array representing the sorted pairwise terms.
    """
    idx = np.argsort(~pair_block_mask(spins, drop_diagonal, drop_off_block), kind='stable')
    return pairs[idx]


def n_pair_same(spins: np.ndarray, drop_diagonal: bool = False, drop_off_block: bool = False) -> int:
    """
    Computes the number of same-spin pairs.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_diagonal: A boolean indicating whether to drop diagonal elements from the mask.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    An integer representing the number of same pairs.
    """
    spins = np.array(spins)
    n_items = spins ** 2
    if drop_diagonal:
        n_items -= spins
        if drop_off_block:
            n_items = n_items / 2
    elif drop_off_block:
        n_items = (n_items - spins) / 2 + spins
    return int(n_items.sum())


def n_pair_diff(spins: np.ndarray, drop_off_block: bool = False) -> int:
    """
    Computes the number of pairs with different spins.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.
    drop_off_block: A boolean indicating whether to drop off-block elements from the mask.

    Returns:
    An integer representing the number of different pairs.
    """
    spins = np.array(spins)
    result = 2 * (spins[:, 0] * spins[:, 1]).sum()
    if drop_off_block:
        result //= 2
    return result


def spin_mask(spins: np.ndarray) -> np.ndarray:
    """
    Computes a mask indicating the spin for each electron.

    Args:
    spins: A 2D array of shape (batch_size, 2) representing the number of spins in each block.

    Returns:
    A 1D array representing the spin mask.
    """
    return np.concatenate([
        np.full((x,), 1-i, dtype=bool)
        for sp in spins
        for i, x in enumerate(sp)
    ])


def spherical_coordinates(x: jax.Array) -> jax.Array:
    """
    Computes the spherical coordinates for a given set of cartesian coordinates.

    Args:
    x: A 2D array of shape (batch_size, 3) representing the cartesian coordinates.

    Returns:
    A tuple of 1D arrays representing the spherical coordinates (r, theta, phi).
    """
    r = jnp.linalg.norm(x, axis=-1)
    x /= jnp.where(r > 1e-6, r, 1)[..., None]
    theta = jnp.arccos(x[..., -1])
    theta = jnp.where(jnp.isnan(theta), 0, theta)

    xy_norm = jnp.linalg.norm(x[..., :2], axis=-1)
    phi = jnp.sign(x[..., 1]) * jnp.arccos(x[..., 0] / xy_norm)
    phi = jnp.where(jnp.isnan(phi), 0, phi)
    return r, theta, phi
