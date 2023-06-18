import heapq
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Sequence, Tuple, Type

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from chex import PRNGKey

from globe.systems.molecule import Atom, Molecule
from globe.systems.property import MolProperty
from globe.systems.scf import Scf
from globe.utils import get_attrs, tree_zip
from globe.utils.config import SystemConfigs
from globe.utils.jax_utils import broadcast


@jax.jit
def reduce_dims(x: jax.Array) -> jax.Array:
    """
    Reduces the dimensions of an array to 1 by taking the mean along all axes except the last.
    """
    while x.ndim > 1:
        x = x.mean(0)
    return x


# A utility class which stores additional information such as the current electron configuration
# and the currrent MCMC proposal width scale.
class MoleculeInstance(Molecule):
    """
    A class representing a molecule instance with additional information such as the current electron configuration
    and the current MCMC proposal width scale.

    Attributes:
        electrons (jax.Array): The electron configuration of the molecule instance.
        restricted (bool): Whether the molecule instance is restricted.
        basis (str): The basis of the molecule instance.
        properties (Tuple[MolProperty, ...]): The properties of the molecule instance.
        _scf (Scf): The self-consistent field of the molecule instance.
    """
    electrons: jax.Array
    restricted: bool
    basis: str
    properties: Tuple[MolProperty, ...]
    _scf: Scf

    def __init__(self,
        electrons: jax.Array,
        atoms: Sequence[Atom],
        spins: Tuple[int, int] | None,
        properties: tuple[MolProperty, ...],
        restricted: bool,
        basis: str) -> None:
        """
        Initializes a MoleculeInstance object.

        Args:
            electrons (jax.Array): The electron configuration of the molecule instance.
            atoms (Sequence[Atom]): The atoms of the molecule instance.
            spins (Optional[Tuple[int, int]]): The spins of the molecule instance.
            properties (Tuple[MolProperty, ...]): The properties of the molecule instance.
            restricted (bool): Whether the molecule instance is restricted.
            basis (str): The basis of the molecule instance.
        """
        super().__init__(atoms, spins)
        assert electrons.shape[-2] == sum(self.spins)
        self.electrons = electrons
        self.properties = properties
        self.restricted = restricted
        self.basis = basis
        self._scf = None
    
    @property
    def scf(self) -> Scf:
        """
        Returns:
            Scf: The self-consistent field of the molecule instance.
        """
        if self._scf is None:
            self._scf = Scf(self, restricted=self.restricted, basis=self.basis).run()
        return self._scf

    @property
    def property_values(self) -> dict:
        """
        Returns:
            dict: The property values of the molecule instance.
        """
        return {
            prop.key: prop.value
            for prop in self.properties
        }
    
    def update(self, electrons: jax.Array, **kwargs) -> None:
        """
        Updates the electron configuration of the molecule instance.

        Args:
            electrons (jax.Array): The new electron configuration of the molecule instance.
            **kwargs: Additional keyword arguments.
        """
        self.electrons = electrons
        for prop in self.properties:
            prop.update(**kwargs)


# We must define these utility functions outside such that they can take advantage of jit
_concat = jax.jit(lambda x: jnp.concatenate(x, axis=-2))
_stack = jax.jit(lambda x: jnp.stack(x))
def batch_molecules(molecules: Sequence[MoleculeInstance], is_sorted: bool = False) -> tuple[jax.Array, jax.Array, SystemConfigs, dict]:
    """
    Batch a sequence of MoleculeInstance objects.

    Args:
        molecules (Sequence[MoleculeInstance]): A sequence of MoleculeInstance objects.
        is_sorted (bool, optional): Whether the molecules are already sorted. Defaults to False.

    Returns:
        Tuple[jax.Array, jax.Array, SystemConfigs, Any]: A tuple containing the batched electrons, atoms, SystemConfigs, and properties.
    """
    if not is_sorted:
        molecules = sorted(molecules)
    electrons, atoms, spins, charges, properties = get_attrs(
        molecules,
        'electrons',
        'positions',
        'spins',
        'charges',
        'property_values'
    )
    return broadcast(_concat(electrons)), _concat(atoms), SystemConfigs(spins, charges), jtu.tree_map(lambda *x: _stack(x), *properties)


_split = jax.jit(jnp.split, static_argnums=1, static_argnames='axis')
def unbatch_electrons(electrons: jax.Array, config: SystemConfigs) -> list[jax.Array]:
    """
    Unbatch electrons to a list of electrons.

    Args:
        electrons (jax.Array): The batched electrons of a molecule instance.
        config (SystemConfigs): The system configurations of the molecule instance.

    Returns:
        list[jax.Array]: The unbatched electrons of the molecule instance.
    """
    return _split(electrons, tuple(np.cumsum(np.sum(config.spins, axis=-1))[:-1]), axis=-2)


def split_to_batches(sequence: Sequence, batch_size: int) -> Generator[Sequence, None, None]:
    """
    Split a sequence into batches of a given size.

    Args:
        sequence (Sequence[jax.Array]): The sequence to be split.
        batch_size (int): The size of each batch.

    Returns:
        Sequence[Sequence[jax.Array]]: A sequence of batches of the given size.
    """
    idx = 0
    while idx < len(sequence):
        yield sequence[idx:idx+batch_size]
        idx += batch_size


class LastBatchBehavior(Enum):
    NONE = 'none'
    FILL_RANDOM = 'fill_random'
    DROP = 'drop'


class DataLoader:
    """
    A interface representing a data loader for a sequence of MoleculeInstance objects.

    Attributes:
        molecules (Sequence[MoleculeInstance]): The sequence of MoleculeInstance objects to be loaded.
        batch_size (int): The size of each batch.
    """
    molecules: Sequence[MoleculeInstance]
    batch_size: int

    def __init__(self, molecules: Sequence[MoleculeInstance], batch_size: int) -> None:
        """
        Initializes a DataLoader object.

        Args:
            molecules (Sequence[MoleculeInstance]): The sequence of MoleculeInstance objects to be loaded.
            batch_size (int): The size of each batch.
        """
        self.molecules = molecules
        self.batch_size = batch_size
    
    def __next__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class RandomLoader(DataLoader):
    """
    A DataLoader that loads MoleculeInstance objects in a random order.

    Attributes:
        order (Sequence[MoleculeInstance]): The sequence of MoleculeInstance objects in random order.
    """
    order: Sequence[MoleculeInstance]
 
    def __init__(self, key: jax.random.PRNGKey, molecules: Sequence[MoleculeInstance], batch_size: int, batch_behavior: LastBatchBehavior) -> None:
        """
        Args:
            key (jax.random.PRNGKey): The key to use for random number generation.
            molecules (Sequence[MoleculeInstance]): The sequence of MoleculeInstance objects to be loaded.
            batch_size (int): The size of each batch.
            batch_behavior (LastBatchBehavior): The behavior to use for the last batch if it is not full.
        """
        super().__init__(molecules, batch_size)
        batch_size = min([batch_size, len(self.molecules)])
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, len(self.molecules))
        idx = np.array(idx).astype(int)
        self.order = []
        for i in idx:
            self.order.append(self.molecules[i])

        residual = len(molecules) % batch_size
        if residual != 0:
            if batch_behavior is LastBatchBehavior.FILL_RANDOM:
                key, subkey = jax.random.split(key)
                to_fill = batch_size - residual
                idx = jax.random.choice(subkey, len(self.molecules) - residual, shape=(to_fill,))
                idx = np.array(idx).astype(int)
                for i in idx:
                    self.order.append(self.molecules[i])
            elif batch_behavior is LastBatchBehavior.DROP:
                self.order = self.order[:-residual]
            elif batch_behavior is LastBatchBehavior.NONE:
                pass

    def __iter__(self) -> Generator[Sequence[MoleculeInstance], None, None]:
        """
        Returns:
            Generator[Sequence[MoleculeInstance]]: A generator that yields batches of MoleculeInstance objects.
        """
        return split_to_batches(self.order, self.batch_size)


@dataclass
class Batch:
    """
    A dataclass representing a batch of MoleculeInstance objects.

    Attributes:
        molecules (Sequence[MoleculeInstance]): The sequence of MoleculeInstance objects in the batch.
    """
    molecules: Sequence[MoleculeInstance]

    def __post_init__(self):
        self.molecules = sorted(self.molecules)

    def to_jax(self) -> jax.Array:
        """
        Returns:
            jax.Array: The JAX array representation of the batch.
        """
        return batch_molecules(self.molecules, is_sorted=True)
    
    @property
    def scfs(self) -> Tuple[jax.Array, ...]:
        """
        Returns:
            Tuple[jax.Array, ...]: The SCF object of each molecule in the batch.
        """
        return tuple(m.scf for m in self.molecules)
    
    @property
    def config(self) -> SystemConfigs:
        """
        Returns:
            SystemConfigs: The SystemConfigs object of the batch.
        """
        return SystemConfigs(*get_attrs(self.molecules, 'spins', 'charges'))

    def update_states(self, electrons: jax.Array, **mol_props: dict) -> None:
        """
        Updates the states of the molecules in the batch.

        Args:
            electrons (jax.Array): The electrons to update the states with.
            **mol_props (Any): The properties to update the states with.
        """
        mol_props = jtu.tree_map(reduce_dims, mol_props)
        split_elec = unbatch_electrons(electrons, self.config)
        assert len(split_elec) == len(self.molecules)
        for m, e, props in tree_zip(self.molecules, unbatch_electrons(electrons, self.config), mol_props):
            m.update(e, **props)
    
    def __getitem__(self, idx: int) -> MoleculeInstance:
        """
        Returns the molecule at the given index.

        Args:
            idx (int): The index of the molecule to return.

        Returns:
            MoleculeInstance: The molecule at the given index.
        """
        return self.molecules[idx]


class Dataset:
    """
    A class representing a dataset of MoleculeInstance objects.

    Attributes:
        molecules (Sequence[MoleculeInstance]): The sequence of MoleculeInstance objects in the dataset.
        batch_size (int): The size of each batch.
        data_loader (DataLoader): The data loader used to load batches from the dataset.
    """
    molecules: Sequence[MoleculeInstance]
    batch_size: int
    data_loader: DataLoader

    def __init__(
        self,
        key: PRNGKey,
        molecules: Sequence[Molecule],
        loader: str,
        batch_behavior: LastBatchBehavior,
        batch_size: int,
        samples_per_molecule: int,
        property_factories: Tuple[Type[MolProperty], ...],
        restricted: bool = True,
        basis: str = 'STO-6G') -> None:
        """
        Initializes a Dataset object.

        Args:
            key (PRNGKey): The random key used to initialize the dataset.
            molecules (Sequence[Molecule]): The sequence of Molecule objects to create MoleculeInstance objects from.
            loader (str): The type of data loader to use. Must be 'random' at the moment.
            batch_behavior (LastBatchBehavior): The behavior of the last batch. Must be a LastBatchBehavior object.
            batch_size (int): The size of each batch.
            samples_per_molecule (int): The number of samples per molecule.
            property_factories (Tuple[Type[MolProperty], ...]): The tuple of MolProperty types to use.
            restricted (bool, optional): Whether or not to use RHF or UHF in SCF. Defaults to True.
            basis (str, optional): The SCF basis to use. Defaults to 'STO-6G'.
        """
        assert loader in ['random']
        batch_behavior = LastBatchBehavior(batch_behavior)
        self.molecules = []
        for m in molecules:
            key, subkey = jax.random.split(key)
            self.molecules.append(MoleculeInstance(
                init_electrons(subkey, m.positions, m.spins, m.charges, samples_per_molecule),
                m.atoms,
                m.spins,
                tuple(fn() for fn in property_factories),
                restricted=restricted,
                basis=basis
            ))
        if loader == 'random':
            key, subkey = jax.random.split(key)
            self.data_loader = RandomLoader(subkey, self.molecules, batch_size, batch_behavior=batch_behavior)
        
        self.iterator = None
        
    def __iter__(self) -> 'Dataset':
        """
        Returns:
            Dataset: The iterator object for the dataset.
        """
        self.iterator = iter(self.data_loader)
        return self
    
    def __next__(self) -> Batch:
        """
        Returns:
            Batch: The next batch in the dataset.
        """
        return Batch(next(self.iterator))


def determine_flips(
    positions: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray
) -> list[int]:
    """
    Determines the atoms for which we swap all spin up and spin down electrons.

    Args:
        positions (jax.Array): The positions of the nuclei.
        alpha (jax.Array): The number of spin up electrons per nucleus.
        beta (jax.Array): The number of spin down electrons per nucleus.

    Returns:
        List[int]: The indices of the nuclei to flip.
    """
    assert (alpha - beta <= 1).all()
    visited = alpha == beta
    distances = np.linalg.norm(
        positions[:, None] - positions,
        axis=-1
    )
    state = np.random.choice(np.where(~visited)[0])
    visited[state] = True
    sequence = [state]
    while not visited.all():
        dists = np.where(visited, np.inf, distances[state])
        state = np.argmin(dists)
        visited[state] = True
        sequence.append(state)
    # by convention alpha > beta, we flip one less for uneven numbers
    to_flip = sequence[1::2]
    return to_flip

def init_electrons(
        key: PRNGKey,
        atoms: np.ndarray,
        spins: Tuple[int, int],
        charges: list[int],
        batch_size: int,
        std: float = 1.0
    ) -> jax.Array:
    """
    Initializes the positions of the electrons in the system.

    Args:
        key (PRNGKey): The key to use for random number generation.
        atoms (jax.Array): The positions of the atoms in the system.
        spins (Tuple[int, int]): The number of spin up and spin down electrons in the system.
        charges (List[int]): The charges of the atoms in the system.
        batch_size (int): The size of the batch.
        std (float, optional): The standard deviation of the normal distribution used to add noise to the positions. Defaults to 1.0.

    Returns:
        jax.Array: The positions of the electrons in the system.
    """
    n_devices = jax.device_count()
    assert batch_size % n_devices == 0
    charges = np.array(charges)
    if sum(charges) != sum(spins):
        p = charges/charges.sum()
        key, subkey = jax.random.split(key)
        atom_idx = jax.random.choice(
            subkey,
            len(charges),
            shape=(batch_size, sum(spins)),
            replace=True,
            p=p)
    else:
        nalpha = np.ceil(charges/2).astype(np.int32)
        nbeta = np.floor(charges/2).astype(np.int32)
        if (sum(nalpha), sum(nbeta)) != spins:
            if spins[0] - spins[1] <= 1:
                flips = determine_flips(atoms, nalpha, nbeta)
                nalpha[flips], nbeta[flips] = nbeta[flips], nalpha[flips]
            else:
                while (sum(nalpha), sum(nbeta)) != spins:
                    key, subkey = jax.random.split(key)
                    i = jax.random.randint(subkey, (), 0, len(nalpha))
                    a, b = nalpha[i], nbeta[i]
                    nalpha[i], nbeta[i] = b, a
        alpha_idx = np.array([
            i for i in range(len(nalpha))
            for _ in range(nalpha[i])
        ], dtype=int)
        beta_idx = np.array([
            i for i in range(len(nbeta))
            for _ in range(nbeta[i])
        ], dtype=int)
        atom_idx = np.concatenate([alpha_idx, beta_idx])
    result = atoms[atom_idx][None].repeat(batch_size, axis=0)
    key, subkey = jax.random.split(key)
    result += jax.random.normal(subkey, shape=result.shape) * std
    return broadcast(result.reshape(n_devices, batch_size//n_devices, -1 , 3))
