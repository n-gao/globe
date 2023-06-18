import functools
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from globe.utils import adj_idx, flatten, nuclei_per_graph
from globe.utils.config import SystemConfigs, group_by_config, inverse_group_idx


@functools.partial(jax.jit, static_argnums=1, static_argnames='config')
def find_axes(atoms: jax.Array, config: SystemConfigs) -> jax.Array:
    """
    Computes the equivariant axes of molecules.

    Args:
    - atoms: a 3D array of shape (n_atoms, 3) containing the coordinates of the atoms.
    - config: a SystemConfigs object containing the configuration of the system.

    Returns:
    - A 3D array of shape (n_graphs, 3, 3) containing the principal axes of the molecule for each graph.
    """
    n_nuclei = nuclei_per_graph(config)
    charges = np.array(list(flatten(config.charges)))
    atoms = jax.lax.stop_gradient(atoms)

    s, axes = get_pca_axis(atoms, config)
    s = jnp.around(s, 5)
    is_ambiguous = jnp.count_nonzero(
        # We must round here to avoid numerical instabilities
        jnp.unique(s, size=3, axis=-1, fill_value=0),
        axis=-1
    ) < jnp.count_nonzero(s, axis=-1)

    # We stretch it twice in case that all three singular values are identical
    pseudo_atoms = get_pseudopositions(atoms, config, 1e-2)
    pseudo_atoms = get_pseudopositions(pseudo_atoms, config, 1e-3)
    pseudo_s, pseudo_axes = get_pca_axis(pseudo_atoms, config)
    pseudo_s = jnp.around(pseudo_s, 5)

    atoms = jnp.where(is_ambiguous.repeat(np.array(n_nuclei))[..., None], pseudo_atoms, atoms)
    s = jnp.where(is_ambiguous[..., None], pseudo_s, s)
    axes = jnp.where(is_ambiguous[..., None, None], pseudo_axes, axes)

    # We sort the negatives instead of reversing the list to keep stable sorting
    order = jnp.argsort(-s, axis=-1)
    axes = jnp.take_along_axis(axes, order[..., None], axis=-2)
    
    # Compute an equivariant vector
    i, j, _ = adj_idx(n_nuclei)
    distances = jnp.linalg.norm(atoms[i] - atoms[j], axis=-1)
    weights = jraph.segment_sum(
        distances,
        np.arange(np.sum(n_nuclei)).repeat(np.repeat(n_nuclei, n_nuclei)),
        num_segments=np.sum(n_nuclei),
        indices_are_sorted=True
    )
    equi_vecs = jraph.segment_mean(
        ((weights * charges)[..., None] * atoms),
        np.arange(len(n_nuclei)).repeat(n_nuclei),
        num_segments=len(n_nuclei),
        indices_are_sorted=True
    )
    # We can expect numerical instabilities up to 1e-4
    # We we are uncertain let's just select the (1,1,1)  vector as reference
    # this is not equivariant but shouldn't matter.
    equi_vecs = jnp.where(jnp.linalg.norm(equi_vecs, axis=-1, keepdims=True) < 1e-3, jnp.ones((3,)), equi_vecs)
    
    ve = jax.vmap(jnp.dot)(equi_vecs, axes)
    flips = ve < 0
    axes = jnp.where(flips[:, None], -axes, axes)

    # Always apply the right hand rule and transpose!
    return jnp.stack([
        axes[..., 0, :],
        axes[..., 1, :],
        jnp.cross(axes[..., 0, :], axes[..., 1, :])
    ], axis=-1)



def get_pca_axis(coords: jax.Array, config: SystemConfigs) -> tuple[jax.Array, jax.Array]:
    """
    Computes the principal axes of the given coordinates.

    Args:
    - coords: An array of shape (n_atoms, 3) representing the coordinates of the atoms.
    - config: A SystemConfigs object containing the charges of the atoms.

    Returns:
    - A tuple containing:
        - s: An array of shape (n_atoms,) representing the eigenvalues of the covariance matrix.
        - axes: An array of shape (n_atoms, 3, 3) representing the eigenvectors of the covariance matrix.
    """
    n_nuclei = nuclei_per_graph(config)
    weights = np.array(list(flatten(config.charges)), dtype=np.float32)
    weights = weights / jraph.segment_sum(
        weights,
        np.arange(len(n_nuclei)).repeat(n_nuclei),
        num_segments=len(n_nuclei)
    ).repeat(np.array(n_nuclei))

    centers = jraph.segment_mean(
        coords*weights[..., None],
        np.arange(len(n_nuclei)).repeat(n_nuclei),
        num_segments=len(n_nuclei),
        indices_are_sorted=True
    )
    coords = coords - jnp.repeat(centers, np.array(n_nuclei), axis=0)
    charges = np.array(list(flatten(config.charges)))
    covs = jraph.segment_mean(
        coords[..., None, :] * coords[..., None] * charges[..., None, None],
        np.arange(len(n_nuclei)).repeat(n_nuclei),
        num_segments=len(n_nuclei),
        indices_are_sorted=True
    )
    s, axes = jnp.linalg.eigh(covs)
    return s, jnp.swapaxes(axes, -1, -2)


def get_projection_vectors(atoms: jax.Array, config: SystemConfigs) -> jax.Array:
    """
    Computes the projection vectors based on the vector inducing the largest coulomb energy.

    Args:
    - atoms: An array of shape (n_atoms, 3) representing the coordinates of the atoms.
    - config: A SystemConfigs object containing the charges of the atoms.

    Returns:
    - An array of shape (n_graphs, 3) representing the projection vectors.
    """
    # Compute pseudo coordiantes based on the vector inducing the largest coulomb energy.
    charges = np.array(list(flatten(config.charges)))
    n_nuclei = nuclei_per_graph(config)
    i, j, _ = adj_idx(n_nuclei, drop_diagonal=True)
    distances = atoms[i] - atoms[j]
    coulomb = charges[i] * charges[j] / jnp.linalg.norm(distances, axis=-1)
    scale_vecs = jnp.concatenate([
        # cou is graphs x edges
        # the argmax is graphs
        # dist is graphs x edges x 3
        (
            jnp.take_along_axis(dist, jnp.argmax(cou, axis=1)[..., None , None], axis=1).squeeze(1)
            if cou.size > 0 else
            jnp.ones((dist.shape[0], 3))
        )
        for cou, dist in zip(
            group_by_config(config, coulomb, lambda _, c: len(c)**2-len(c)),
            group_by_config(config, distances, lambda _, c: len(c)**2-len(c))
        )
    ])
    scale_vecs = scale_vecs[inverse_group_idx(config)]
    scale_vecs /= jnp.linalg.norm(scale_vecs, axis=-1, keepdims=True)
    return scale_vecs


def get_pseudopositions(atoms: jax.Array, config: SystemConfigs, eps: float = 1e-2) -> jax.Array:
    """
    Computes the pseudo-coordinates of the atoms based on the vector inducing the largest coulomb energy.

    Args:
    - atoms: An array of shape (n_atoms, 3) representing the coordinates of the atoms.
    - config: A SystemConfigs object containing the charges of the atoms.
    - eps: A float representing the scaling factor for the projection vectors.

    Returns:
    - An array of shape (n_atoms, 3) representing the pseudo-coordinates of the atoms.
    """
    n_nuclei = nuclei_per_graph(config)
    # Compute pseudo coordiantes based on the vector inducing the largest coulomb energy.
    scale_vecs = get_projection_vectors(atoms, config)
    # Projected atom positions
    scale_vecs = scale_vecs.repeat(np.array(n_nuclei), axis=0)
    proj = jax.vmap(jnp.dot)(atoms, scale_vecs)[..., None] * scale_vecs
    pseudo_atoms = proj * eps + atoms
    return pseudo_atoms


def rotate_by_svd(x):
    """
    Applies a rotation to the input array using the Singular Value Decomposition (SVD) method.

    Args:
    - x: An array of shape (n, m) representing the input array.

    Returns:
    - An array of shape (n, m) representing the rotated array.
    """
    l, s, u = np.linalg.svd(x)
    return x@u.T


@functools.partial(jax.jit, static_argnums=1, static_argnames='config')
def find_equi_vecs(atoms: jax.Array, config: SystemConfigs) -> jax.Array:
    """
    Computes the equivariant vectors for each atom in the input array based on the Coulomb energy between atoms.

    Args:
    - atoms: An array of shape (n_atoms, 3) representing the coordinates of the atoms.
    - config: A SystemConfigs object containing the charges of the atoms.

    Returns:
    - An array of shape (n_atoms, 3) representing the equivariant vectors for each atom in the input array.
    """
    n_nuc = nuclei_per_graph(config)
    diag = np.concatenate([np.eye(n).reshape(-1) for n in n_nuc])
    flat_charges = jnp.array(np.concatenate(config.charges))
    idx_i, idx_j, _ = adj_idx(n_nuc)
    diffs = atoms[idx_i] - atoms[idx_j]
    dists = jnp.linalg.norm(diffs, axis=-1)
    coulomb = flat_charges[idx_i] * flat_charges[idx_j] * jnp.exp(-dists) * (1-diag)
    coulomb /= jraph.segment_sum(
        coulomb,
        idx_i,
        sum(n_nuc)
    )[idx_i]
    vecs = coulomb[..., None] * atoms[idx_j]
    vecs = jraph.segment_sum(
        vecs,
        idx_i,
        sum(n_nuc)
    ) - atoms
    vecs /= jnp.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs


@jax.jit
def get_rotation_vec(vec1: jax.Array, vec2: jax.Array) -> tuple[jax.Array, float, float]:
    """
    Computes the rotation vector, sine and cosine of the angle between two input vectors.

    Args:
    - vec1: An array of shape (3,) representing the first vector.
    - vec2: An array of shape (3,) representing the second vector.

    Returns:
    - A tuple containing:
        - An array of shape (3,) representing the rotation vector.
        - A float representing the sine of the angle between the two input vectors.
        - A float representing the cosine of the angle between the two input vectors.
    """
    vec1 /= jnp.linalg.norm(vec1)
    vec2 /= jnp.linalg.norm(vec2)
    rot_vec = jnp.cross(vec1, vec2)
    sin_angle = jnp.linalg.norm(rot_vec)
    rot_vec /= sin_angle
    cos_angle = jnp.dot(vec1, vec2)
    return rot_vec, sin_angle, cos_angle


@jax.jit
def rotate_by_vec(v: jax.Array, rot_vec: jax.Array, sin_theta: float, cos_theta: float) -> jax.Array:
    """
    Rotates a vector `v` by a rotation vector `rot_vec` using the provided sine and cosine of the angle between the two vectors.

    Args:
    - v: An array of shape (3,) representing the vector to be rotated.
    - rot_vec: An array of shape (3,) representing the rotation vector.
    - sin_theta: A float representing the sine of the angle between the two input vectors.
    - cos_theta: A float representing the cosine of the angle between the two input vectors.

    Returns:
    - An array of shape (3,) representing the rotated vector.
    """
    rotated_vec = v*cos_theta + jnp.cross(rot_vec, v) * sin_theta + rot_vec * jnp.dot(rot_vec, v) * (1 - cos_theta)
    # Check if the angle is 0° in that case rot_vec will be a zero vector.
    rotated_vec = jnp.where(cos_theta > 1 - 1e-3, v, rotated_vec)
    # Check if the angle is 180° in that case rot_vec will also be a zero vector.
    rotated_vec = jnp.where(cos_theta < -(1 - 1e-3), -v, rotated_vec)
    return rotated_vec


@jax.jit
def rotate_unit_frame(equi_vec: jax.Array) -> jax.Array:
    """
    Rotates the unit frame of a given vector to align with the x-axis.

    Args:
    - equi_vec: An array of shape (3,) representing the vector to be rotated.

    Returns:
    - An array of shape (3, 3) representing the rotated unit frame.
    """
    rotvec, sin_theta, cos_theta = get_rotation_vec(equi_vec, jnp.array([1, 0, 0]))
    rot_frame_by_vec = jax.vmap(rotate_by_vec, in_axes=(0, None, None, None))
    frames = rot_frame_by_vec(jnp.eye(3), rotvec, sin_theta, cos_theta)
    return frames


@functools.partial(jax.jit, static_argnums=1, static_argnames='config')
def find_atom_frames(atoms: jax.Array, config: SystemConfigs) -> jax.Array:
    """
    Finds the frames of reference for each atom.

    Args:
    - atoms: An array of shape (N, 3) representing the coordinates of the atoms in the system.
    - config: An instance of SystemConfigs containing the configuration parameters for the system.

    Returns:
    - An array of shape (N, 3, 3) representing the frames of reference for each atom in the system.
    """
    equi_vecs = find_equi_vecs(atoms, config)
    atom_frames = jax.vmap(rotate_unit_frame)(equi_vecs)
    return atom_frames
