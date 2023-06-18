"""
This file contains two classes: Atom and Molecule.
An atom consists of an element and coordinates while a molecule
is composed by a set of atoms.

The classes contain simple logic functions to obtain spins, charges
and coordinates for molecules.
"""
import math
import numbers
from collections import Counter
from functools import cached_property, total_ordering
import re
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import pyscf
from globe.systems.constants import ANGSTROM_TO_BOHR
from globe.systems.element import (ELEMENT_BY_ATOMIC_NUM, ELEMENT_BY_SYMBOL,
                                     Element)
from globe.utils import argsort, comp_clouds, itemgetter


Symbol = str | int


class Atom:
    element: Element
    position: np.ndarray
    
    def __init__(self, symbol: Symbol, position: Tuple[float, float, float] = (0, 0, 0), units='bohr') -> None:
        assert units in ['bohr', 'angstrom']
        if isinstance(symbol, str):
            self.element = ELEMENT_BY_SYMBOL[symbol]
        elif isinstance(symbol, numbers.Integral):
            self.element = ELEMENT_BY_ATOMIC_NUM[symbol]
        else:
            raise ValueError()
        if position is None:
            position = (0, 0, 0)
        assert len(position) == 3
        self.position = np.array(position)
        if units == 'angstrom':
            self.position *= ANGSTROM_TO_BOHR

    @property
    def atomic_number(self):
        return self.element.atomic_number

    @property
    def symbol(self):
        return self.element.symbol

    def __str__(self):
        return self.element.symbol
    
    def __repr__(self):
        return f'{self.element.symbol} {str(self.position)}'
    
    @staticmethod
    def from_repr(rep):
        symbol = rep.split(' ')[0]
        position = ' '.join(rep.split(' ')[1:])
        position = re.findall(r'([+-]?[0-9]+([.][0-9]*)?|[.][0-9]+)', position)
        position = [float(p[0]) for p in position]
        return Atom(symbol, position)


@total_ordering
class Molecule:
    atoms: Tuple[Atom]
    _spins: Optional[Tuple[int, int]]
    
    def __init__(self, atoms: Sequence[Atom], spins: Optional[Tuple[int, int]] = None) -> None:
        self.atoms = tuple(atoms)
        # Sort atoms by charge
        self.atoms = itemgetter(*argsort(self.charges))(atoms)
        del self.charges # clear cache
        self._spins = spins

    @cached_property
    def charges(self):
        return tuple(a.atomic_number for a in self.atoms)

    @cached_property
    def np_positions(self):
        positions = np.array([a.position for a in self.atoms], dtype=np.float32)
        positions -= positions.mean(0, keepdims=True)
        return positions

    @cached_property
    def positions(self):
        return jnp.array(self.np_positions, dtype=jnp.float32)

    @cached_property
    def spins(self):
        if self._spins is not None:
            return self._spins
        else:
            n_electrons = sum(self.charges)
            return (math.ceil(n_electrons/2), math.floor(n_electrons/2))
    
    def to_pyscf(self, basis='STO-6G', verbose: int = 3):
        mol = pyscf.gto.Mole(atom=[
            [a.symbol, p]
            for a, p in zip(self.atoms, self.np_positions)
        ], unit='bohr', basis=basis, verbose=verbose)
        mol.spin = self.spins[0] - self.spins[1]
        mol.charge = sum(self.charges) - sum(self.spins)
        mol.build()
        return mol

    def __str__(self) -> str:
        result = ''
        if len(self.atoms) == 1:
            result = str(self.atoms[0])
        else:
            vals = dict(Counter(str(a) for a in self.atoms))
            result = ''.join(str(key) + (str(val) if val > 1 else '') for key, val in vals.items())
        if sum(self.spins) < sum(self.charges):
            result += 'plus'
        elif sum(self.spins) > sum(self.charges):
            result += 'minus'
        return result
    
    def __repr__(self) -> str:
        atoms = '\n'.join(map(repr, self.atoms))
        return f'Spins: {self.spins}\n{atoms}'
    
    @staticmethod
    def from_repr(rep):
        return Molecule([Atom.from_repr(r) for r in rep.split('\n')[1:]])

    def __lt__(self, other):
        return (sum(self.spins), self.spins, self.charges) < (sum(other.spins), other.spins, other.charges)
    
    def __eq__(self, other):
        # While this does not formally meet the requirement for a equality
        # we implement it to get stable sorting.
        return (self.spins, self.charges) == (other.spins, other.charges)

    def equivalent(self, other):
        if self != other:
            return False
        return comp_clouds(self.np_positions, other.np_positions, self.charges, other.charges)

    def __hash__(self):
        return hash((self.spins, self.charges))
