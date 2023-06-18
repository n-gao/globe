import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from globe.systems.molecule import Atom, Molecule, Symbol


def atomic(symbol: Symbol) -> Molecule:
    return Molecule([
        Atom(symbol, (0, 0, 0)),
    ])


def diatomic(symbol1: Symbol, symbol2: Symbol, distance: float) -> Molecule:
    return Molecule([
        Atom(symbol1, (-distance/2, 0, 0)),
        Atom(symbol2, (distance/2, 0, 0)),
    ])


def chain(symbol: Symbol, n: int, distance: float) -> Molecule:
    return Molecule([
        Atom(symbol, (distance*i - (distance*n-1)/2, 0, 0))
        for i in range(n)
    ])


def rectangle(symbol: Symbol, theta: float, R: float) -> Molecule:
    y = np.sin(np.radians(theta/2)) * R
    x = np.cos(np.radians(theta/2)) * R
    return Molecule([
        Atom(symbol, (x, y, 0.0)),
        Atom(symbol, (x, -y, 0.0)),
        Atom(symbol, (-x, y, 0.0)),
        Atom(symbol, (-x, -y, 0.0))
    ])


def regular_polygon(symbol: Symbol, n: int, R: float) -> Molecule:
    return Molecule([
        Atom(symbol, (
            np.sin(i/n * 2 * np.pi) * R,
            np.cos(i/n * 2 * np.pi) * R,
            0.0
        ))
        for i in range(n)
    ])


def cyclobutadiene(state: str = None, alpha: float = None) -> Molecule:
    """
    Returns a Molecule object representing cyclobutadiene in the specified state.

    Args:
    - state (str): The state of the cyclobutadiene. Can be either 'ground' or 'transition'.
    - alpha (float): A float between 0 and 1 representing the interpolation between the two states.

    Returns:
    - A Molecule object representing cyclobutadiene in the specified state.
    """
    # https://github.com/deepmind/ferminet/blob/jax/ferminet/configs/organic.py
    assert state is not None or alpha is not None
    if state == 'ground':
        return Molecule([
            Atom('C', (0.0000000e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.9555318e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.9555318e+00, 2.5586891e+00, 0.0000000e+00)),
            Atom('C', (0.0000000e+00, 2.5586891e+00, 0.0000000e+00)),
            Atom('H', (-1.4402903e+00, -1.4433100e+00, 1.7675451e-16)),
            Atom('H', (4.3958220e+00, -1.4433100e+00, -1.7675451e-16)),
            Atom('H', (4.3958220e+00, 4.0019994e+00, 1.7675451e-16)),
            Atom('H', (-1.4402903e+00, 4.0019994e+00, -1.7675451e-16)),
        ])
    elif state == 'transition':
        return Molecule([
            Atom('C', (0.0000000e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.7419927e+00, 0.0000000e+00, 0.0000000e+00)),
            Atom('C', (2.7419927e+00, 2.7419927e+00, 0.0000000e+00)),
            Atom('C', (0.0000000e+00, 2.7419927e+00, 0.0000000e+00)),
            Atom('H', (-1.4404647e+00, -1.4404647e+00, 1.7640606e-16)),
            Atom('H', (4.1824574e+00, -1.4404647e+00, -1.7640606e-16)),
            Atom('H', (4.1824574e+00, 4.1824574e+00, 1.7640606e-16)),
            Atom('H', (-1.4404647e+00, 4.1824574e+00, -1.7640606e-16))
        ])
    else:
        m1 = cyclobutadiene('ground').np_positions
        m2 = cyclobutadiene('transition').np_positions
        return Molecule([
            Atom(c, p)
            for c, p in zip('CCCCHHHH', m1*alpha + (1-alpha)*m2)
        ])


def ammonia_gs() -> Molecule:
    return Molecule([
        Atom('N', (0, 0, 0.22013)),
        Atom('H', (0, 1.77583, -0.51364)),
        Atom('H', (1.53791, -0.88791, -0.51364)),
        Atom('H', (-1.53791, -0.88791, -0.51364)),
    ])


def methane_gs() -> Molecule:
    return Molecule([
        Atom('C', (0.00000, 0.00000, 0.00000)),
        Atom('H', (1.18886, 1.18886, 1.18886)),
        Atom('H', (-1.18886, -1.18886, 1.18886)),
        Atom('H', (1.18886, -1.18886, -1.18886)),
        Atom('H', (-1.18886, 1.18886, -1.18886))
    ])


def ethene_gs() -> Molecule:
    return Molecule([
        Atom('C', (0.00000, 0.00000, 1.26135)),
        Atom('C', (0.00000, 0.00000, -1.26135)),
        Atom('H', (0.00000, 1.74390, 2.33889)),
        Atom('H', (0.00000, -1.74390, 2.33889)),
        Atom('H', (0.00000, 1.74390, -2.33889)),
        Atom('H', (0.00000, -1.74390, -2.33889))
    ])


def bicyclobutane() -> Molecule:
    return Molecule([
        Atom('C', (0.0, 2.13792, 0.58661)),
        Atom('C', (0.0, -2.13792, 0.58661)),
        Atom('C', (1.41342, 0.0, -0.58924)),
        Atom('C', (-1.41342, 0.0, -0.58924)),
        Atom('H', (0.0, 2.33765, 2.64110)),
        Atom('H', (0.0, 3.92566, -0.43023)),
        Atom('H', (0.0, -2.33765, 2.64110)),
        Atom('H', (0.0, -3.92566, -0.43023)),
        Atom('H', (2.67285, 0.0, -2.19514)),
        Atom('H', (-2.67285, 0.0, -2.19514))
    ])


def benzene() -> Molecule:
    return Molecule([
        Atom('C', (0.00000, 2.63664, 0.00000)),
        Atom('C', (2.28339, 1.31832, 0.00000)),
        Atom('C', (2.28339, -1.31832, 0.00000)),
        Atom('C', (0.00000, -2.63664, 0.00000)),
        Atom('C', (-2.28339, -1.31832, 0.00000)),
        Atom('C', (-2.28339, 1.31832, 0.00000)),
        Atom('H', (0.00000, 4.69096, 0.00000)),
        Atom('H', (4.06250, 2.34549, 0.00000)),
        Atom('H', (4.06250, -2.34549, 0.00000)),
        Atom('H', (0.00000, -4.69096, 0.00000)),
        Atom('H', (-4.06250, -2.34549, 0.00000)),
        Atom('H', (-4.06250, 2.34549, 0.00000))
    ])


def benzene_dimer(distance: float) -> Molecule:
    """
    Create a benzene dimer with a given distance between the two rings.
    The rings are rotated by 90 degrees with respect to each other.
    """
    ben_mol = benzene()
    charges = ben_mol.charges
    ben_pos = ben_mol.np_positions
    # Rotate by 90 degrees
    ben_pos2 = ben_pos[:, (0, 2, 1)]
    ben_pos2 -= np.array([0, distance, 0])
    return Molecule([
        Atom(c, pos)
        for c, pos in zip(np.tile(charges, 2), np.concatenate([ben_pos, ben_pos2], axis=0))
    ])


def deeperwin_mol(name, state, base_dir='.', return_cfgs=False) -> list[Molecule] | tuple[list[Molecule], dict[str, Any]]:
    """
    Load a molecule from the deeperwin dataset.

    Args:
    - name: Name of the molecule
    - state: State of the molecule
    - base_dir: Base directory of the data
    - return_cfgs: Whether to return the configuration dictionary
    """
    if name == 'Ethene':
        charges = 'CCHHHH'
    elif name == 'Methane':
        charges = 'CHHHH'
    else:
        charges = 'H' * 10
    try:
        cfg_file = os.path.join(base_dir, f'data/{state}_{name}.yml')
        with open(cfg_file) as inp:
            cfg = yaml.safe_load(inp)
    except FileNotFoundError:
        cfg_file = os.path.join(Path(__file__).parents[2], f'data/{state}_{name}.yml')
        with open(cfg_file) as inp:
            cfg = yaml.safe_load(inp)
    result = []
    for conf in cfg['changes']:
        result.append(Molecule([
            Atom(c, r)
            for c, r in zip(charges, conf['R'])
        ]))
    if not return_cfgs:
        return result
    else:
        return result, cfg['changes']


def distanced(conf: dict, distance: float) -> Molecule:
    """
    Returns a molecule with two copies of the given configuration separated by distance.

    Args:
    - conf: A configuration dictionary.
    - distance: The distance between the two copies.
    """
    mol = get_molecules([conf])[0]
    return Molecule([
        Atom(atom.symbol, atom.position - np.array([distance/2, 0, 0]))
        for atom in mol.atoms
    ] + [
        Atom(atom.symbol, atom.position + np.array([distance/2, 0, 0]))
        for atom in mol.atoms
    ])


def get_molecules(systems: list[tuple[Any, ...]]) -> list[Molecule]:
    """
    Returns a list of Molecule objects from a list of configurations.

    Args:
    - systems: A list of system objects.

    Returns:
    - A list of Molecule objects.
    """
    result = []
    for s in systems:
        if isinstance(s, dict):
            mol = globals()[s['name']](**s['config'])
        elif isinstance(s, (tuple, list)):
            mol = globals()[s[0]](*s[1:])
        else:
            raise RuntimeError()
        if isinstance(mol, (tuple, list)):
            result = result + list(mol)
        else:
            result.append(mol)
    return sorted(result)
