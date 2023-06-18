import logging
import os
from collections import OrderedDict, defaultdict
from typing import Callable, Tuple

import h5py
import jax
import jax.tree_util as jtu
import numpy as np
import pyscf
from scipy.optimize import linear_sum_assignment, minimize

from globe.nn.coords import find_axes
from globe.nn.orbitals import get_orbitals
from globe.systems.molecule import Molecule
from globe.utils.config import SystemConfigs

get_orbitals = jax.jit(get_orbitals, static_argnums=1, static_argnames='config')


class Scf:
    """
    A Hartree-Fock mean-field solver for molecules.

    Attributes:
    - molecule (Molecule): The molecule to solve.
    - _mol (pyscf.gto.Mole): The PySCF molecule object.
    - _mean_field (pyscf.scf.hf.SCF): The PySCF mean-field object.
    - restricted (bool): Whether the calculation is restricted or unrestricted.
    - _coeff (np.ndarray): The canonicalized molecular orbital coefficients.

    Methods:
    - __init__(self, molecule: Molecule, restricted: bool = True, basis='STO-6G', verbose=3) -> None:
        Initializes the Scf object.
    - run(self, chkfile: str = None):
        Runs the Hartree-Fock calculation.
    - eval_molecular_orbitals(self, electrons: jax.Array, deriv: bool = False) -> Tuple[jax.Array, jax.Array]:
        Evaluates the molecular orbitals for a given set of electrons.
    - energy(self):
        Returns the Hartree-Fock energy.
    - mo_coeff(self):
        Returns the molecular orbital coefficients.
    """
    molecule: Molecule
    _mol: pyscf.gto.Mole
    _mean_field: pyscf.scf.hf.SCF
    restricted: bool
    _coeff: np.ndarray = None

    def __init__(self, molecule: Molecule, restricted: bool = True, basis='STO-6G', verbose=3) -> None:
        """
        Args:
        - molecule (Molecule): The molecule to solve.
        - restricted (bool): Whether the calculation is restricted or unrestricted.
        - basis (str): The basis set to use.
        - verbose (int): The verbosity level.
        """
        self.molecule = molecule
        self.restricted = restricted
        self._mol = self.molecule.to_pyscf(basis, verbose)
        if restricted:
            self._mean_field = pyscf.scf.RHF(self._mol)
        else:
            self._mean_field = pyscf.scf.UHF(self._mol)
    
    def run(self, chkfile: str = None):
        """
        Runs the Hartree-Fock calculation.

        Args:
        - chkfile (str): The checkpoint file to use.
        """
        self._mean_field.chkfile = chkfile
        if chkfile is not None and os.path.exists(chkfile):
            with h5py.File(chkfile, 'r') as inp:
                self._mean_field.mo_coeff = inp['scf']['mo_coeff'][()]
                self._mean_field.e_tot = inp['scf']['e_tot'][()]
                logging.info(f'Loaded HF energy: {self.energy}')
        else:
            self._mean_field.kernel()
        if self.restricted:
            self._coeff = canonicalize_weights(self)
        return self

    @property
    def energy(self):
        return self._mean_field.e_tot
    
    @property
    def mo_coeff(self):
        if self._coeff is not None:
            return self._coeff
        if self.restricted:
            coeffs = (self._mean_field.mo_coeff,)
        else:
            coeffs = self._mean_field.mo_coeff
        return np.array(coeffs)
    
    def eval_molecular_orbitals(self, electrons: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Evaluates the molecular orbitals for a given set of electrons.

        Args:
        - electrons (jax.Array): A 2D array of shape (n_electrons, 3) containing the electron coordinates.

        Returns:
        - Tuple[jax.Array, jax.Array]: A tuple containing the molecular orbitals.
        """
        if self._mol.cart:
            raise NotImplementedError(
                'Evaluation of molecular orbitals using cartesian GTOs.')

        gto_op = 'GTOval_sph'
        electrons = np.array(electrons)
        ao_values = self._mol.eval_gto(gto_op, electrons)
        mo_values = tuple(np.matmul(ao_values, coeff) for coeff in self.mo_coeff)
        if self.restricted:
            mo_values *= 2
        return mo_values


def get_number_ao(scf: Scf) -> tuple[int]:
    """
    Returns the number of atomic orbitals for each atom.

    Args:
    - scf: An instance of the Scf class.

    Returns:
    - Tuple[int]: A tuple containing the number of atomic orbitals for each atom.
    """
    counts = defaultdict(int)
    for l in scf._mol.ao_labels():
        counts[int(l.split(' ')[0])] += 1
    return tuple(
        counts[i]
        for i in range(max(counts) + 1)
    )


def make_mask(nuc_orbitals: tuple[int], nuc_assign: list[Tuple[int, int]]) -> np.ndarray:
    """
    Creates a boolean mask for the given nuclear orbitals and assignments.

    Args:
    - nuc_orbitals (Tuple[int]): A tuple containing the number of atomic orbitals for each atom.
    - nuc_assign (List[Tuple[int, int]]): A list of tuples containing the pairs of atoms in each orbital.

    Returns:
    - np.ndarray: A boolean mask for the given nuclear orbitals and assignments.
    """
    segments = np.cumsum((0,) + nuc_orbitals + (0,))
    mask = np.zeros((segments[-1], len(nuc_assign)), dtype=bool)
    for k, (i, j) in enumerate(nuc_assign):
        mask[segments[i]:segments[i+1], k] = True
        mask[segments[j]:segments[j+1], k] = True
    return mask


def get_ao_scores(nuc_orbitals: tuple[int]) -> np.ndarray:
    """
    Returns the weights for each atomic orbital.

    Args:
    - nuc_orbitals (Tuple[int]): A tuple containing the number of atomic orbitals for each atom.

    Returns:
    - np.ndarray: An array containing the scores of the atomic orbitals.
    """
    result = np.concatenate([
        np.arange(1, n+1)
        for n in nuc_orbitals
    ])
    return result


def get_mo_scores(types: np.ndarray) -> np.ndarray:
    """
    Returns the weights for each molecular orbital.

    Args:
    - types (jax.Array): An array containing the types of each molecular orbital.

    Returns:
    - np.ndarray: An array containing the scores of the molecular orbitals.
    """
    types = np.array(types)
    result = np.zeros_like(types)
    for i, v in enumerate(np.unique(types)):
        result[types == v] = i
    return result.max() - result + 1


def make_target(nuc_orbitals: tuple[int], nuc_assign: list[tuple[int, int]], types: np.ndarray) -> np.ndarray:
    """
    Creates a boolean array indicating which molecular orbitals correspond to each atomic orbital.

    Args:
    - nuc_orbitals (Tuple[int]): A tuple containing the number of atomic orbitals for each atom.
    - nuc_assign (List[Tuple[int, int]]): A list of tuples containing the indices of the atoms involved in each molecular orbital.
    - types (jax.Array): An array containing the types of each molecular orbital.

    Returns:
    - np.ndarray: A boolean array indicating which molecular orbitals correspond to each atomic orbital.
    """
    nuc_prio = defaultdict(lambda: defaultdict(list))
    for i, (a, t) in enumerate(zip(nuc_assign, types)):
        for x in np.unique(a):
            nuc_prio[x][t].append(i)
    nuc_prio = OrderedDict({k: OrderedDict(v) for k, v in nuc_prio.items()})
    offsets = np.cumsum([0, *nuc_orbitals[:-1]])
    result = np.zeros((np.sum(nuc_orbitals), len(nuc_assign)), dtype=bool)

    for atom, order in nuc_prio.items():
        off = offsets[atom]
        for vals in order.values():
            for i in range(len(vals)):
                result[off+i, vals] = True
            off += len(vals)
    return result


def to_mat(x: np.ndarray) -> np.ndarray:
    """
    Reshapes the input array into a matrix and scales it to have determinant 1.

    Args:
    - x (np.ndarray): An array to be reshaped.

    Returns:
    - np.ndarray: A matrix with determinant 1.
    """
    mat = x.reshape(int(np.sqrt(x.size)), -1)
    a = np.abs(np.linalg.det(mat)) ** (1/mat.shape[0])
    return mat/a


def make_minbasis_loss(coeff: np.ndarray, target: np.ndarray) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """
    Creates to loss functions for minimizing the difference between a predicted matrix and a target matrix.
    One for direct optimization and the second produces a penalty matrix for all possible permutations of the rows and columns of the input matrix..

    Args:
    - coeff (jax.Array): A matrix of coefficients.
    - target (np.ndarray): A boolean array indicating which molecular orbitals correspond to each atomic orbital.

    Returns:
    - Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]: A tuple containing two functions:
        - loss: A function that takes a matrix as input and returns a float representing the loss.
        - perm_loss: A function that takes a matrix as input and returns a matrix of losses for all possible permutations of the rows and columns of the input matrix.
    """
    def loss(x: jax.Array) -> float:
        mat = to_mat(x)
        pred = coeff@mat
        result = (pred[~target]**2).sum()
        result += ((1-np.linalg.norm(np.where(target, pred, 0), axis=-2))**2).sum()
        return result
    
    def perm_loss(x: jax.Array) -> np.ndarray:
        mat = to_mat(x)
        pred = coeff@mat
        n = coeff.shape[-1]
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                result[i, j] = ((pred[:, i][~target[:, j]])**2).sum()
                result[i, j] += (1 - np.linalg.norm(pred[:, i][target[:, j]]))**2
        return result
    
    return loss, perm_loss


def make_generic_loss(coeff: jax.Array, mask: np.ndarray, ao_scores: np.ndarray, mo_scores: np.ndarray, score_weight: float = 0) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """
    Creates two loss functions for minimizing the difference between a predicted matrix and a target matrix.
    One for direct optimization and the second produces a penalty matrix for all possible permutations of the rows and columns of the input matrix.

    This function works for arbitrary bases and is not limited to the minimal basis. But results in worse performance.

    Args:
    - coeff (jax.Array): A matrix of coefficients.
    - mask (np.ndarray): A boolean array indicating which molecular orbitals correspond to each atomic orbital.
    - ao_scores (np.ndarray): A 1D array of scores for each atomic orbital.
    - mo_scores (np.ndarray): A 1D array of scores for each molecular orbital.
    - score_weight (float): A weight for the penalty term in the loss function.

    Returns:
    - Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]: A tuple containing two functions:
        - loss: A function that takes a matrix as input and returns a float representing the loss.
        - perm_loss: A function that takes a matrix as input and returns a matrix of losses for all possible permutations of the rows and columns of the input matrix.
    """
    penalty_mask = np.where(mask, ao_scores[..., None] * mo_scores, 10000)
    def loss(x: np.ndarray) -> float:
        mat = to_mat(x)
        pred = coeff@mat
        result = score_weight * np.sum(penalty_mask * np.abs(pred))
        result += ((pred * ~mask)**2).sum()
        result += 1000 * ((1 - np.linalg.norm(pred, axis=0))**2).sum()
        return result

    def perm_loss(x: np.ndarray) -> np.ndarray:
        mat = to_mat(x)
        n = mat.shape[-1]
        def compute_loss(i, j):
            test = np.copy(mat)
            test[:, [j, i]] = test[:, [i, j]]
            return loss(test)

        result = np.vectorize(compute_loss)(*np.where(np.ones((n, n)))).reshape(n, n)
        return result
    return loss, perm_loss


def canonicalize_weights(scf: Scf, maxiter: int = 10) -> np.ndarray:
    """
    Canonicalizes the molecular orbital coefficients of a Hartree-Fock calculation.

    Args:
    - scf (Scf): A Hartree-Fock calculation object.
    - maxiter (int): The maximum number of iterations to perform.

    Returns:
    - np.ndarray: A matrix of canonical molecular orbital coefficients.
    """
    orbitals = get_number_ao(scf)
    conf = SystemConfigs((scf.molecule.spins,), (scf.molecule.charges,))
    axes = find_axes(scf.molecule.positions, conf)
    axes = np.array(axes).reshape(3, 3)
    _, orb_type, orb_assoc, _ = jtu.tree_map(np.array, get_orbitals(
        scf.molecule.positions@axes,
        conf
    ))
    types = orb_type.tolist()
    nuc_idx = orb_assoc.tolist()
    n = (scf._mean_field.mo_occ > 0).sum()
    coeff = scf.mo_coeff[0, :, :n]
    assert len(coeff.shape) == 2

    # New generic calc
    # mask = make_mask(orbitals, nuc_idx)
    # ao_scores = get_ao_scores(orbitals)
    # mo_scores = get_mo_scores(types)
    # score_mask = ao_scores[::-1][:, None] * mask
    # loss, perm_loss = make_new_loss(coeff, mask, ao_scores, mo_scores)
    
    # Minimal basis calc
    target = make_target(orbitals, nuc_idx, types)
    score_mask = np.arange(target.shape[0])[::-1][:, None] * target
    loss, perm_loss = make_minbasis_loss(coeff, target)

    mat = np.eye(n)
    best_loss = np.inf
    for i in range(maxiter):
        # find optimal permutation to minimize initial loss
        perm = linear_sum_assignment(perm_loss(mat))[1]
        init = mat[..., perm].reshape(-1)
        # Minimize objective
        x = minimize(loss, init)
        mat = np.array(to_mat(x.x))
        # Check for convergence
        if np.abs(best_loss - x.fun) < 1e-5:
            break
        else:
            best_loss = x.fun
            if i == maxiter:
                raise RuntimeError("Reached maxiter.")
    # Align signs
    result = coeff@mat
    flips = np.sign((result * score_mask).sum(0))
    result *= flips
    return np.concatenate([
        result[None],
        scf.mo_coeff[..., n:]
    ], axis=-1)
