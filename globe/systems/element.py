# https://github.com/deepmind/ferminet/blob/jax/ferminet/utils/elements.py
# This file is largely taken from Spencer et al., 2020.
from typing import NamedTuple

import numpy as np


class Element(NamedTuple):
    atomic_number: int
    symbol: str
    period: int
    spin: int = 0


# Static array of all relevant elements.
_ELEMENTS = (
    Element(symbol='H', atomic_number=1, period=1),
    Element(symbol='He', atomic_number=2, period=1),
    Element(symbol='Li', atomic_number=3, period=2),
    Element(symbol='Be', atomic_number=4, period=2),
    Element(symbol='B', atomic_number=5, period=2),
    Element(symbol='C', atomic_number=6, period=2),
    Element(symbol='N', atomic_number=7, period=2),
    Element(symbol='O', atomic_number=8, period=2),
    Element(symbol='F', atomic_number=9, period=2),
    Element(symbol='Ne', atomic_number=10, period=2),
    Element(symbol='Na', atomic_number=11, period=3),
    Element(symbol='Mg', atomic_number=12, period=3),
    Element(symbol='Al', atomic_number=13, period=3),
    Element(symbol='Si', atomic_number=14, period=3),
    Element(symbol='P', atomic_number=15, period=3),
    Element(symbol='S', atomic_number=16, period=3),
    Element(symbol='Cl', atomic_number=17, period=3),
    Element(symbol='Ar', atomic_number=18, period=3),
    Element(symbol='K', atomic_number=19, period=4),
    Element(symbol='Ca', atomic_number=20, period=4),
    Element(symbol='Sc', atomic_number=21, period=4, spin=1),
    Element(symbol='Ti', atomic_number=22, period=4, spin=2),
    Element(symbol='V', atomic_number=23, period=4, spin=3),
    Element(symbol='Cr', atomic_number=24, period=4, spin=6),
    Element(symbol='Mn', atomic_number=25, period=4, spin=5),
    Element(symbol='Fe', atomic_number=26, period=4, spin=4),
    Element(symbol='Co', atomic_number=27, period=4, spin=3),
    Element(symbol='Ni', atomic_number=28, period=4, spin=2),
    Element(symbol='Cu', atomic_number=29, period=4, spin=1),
    Element(symbol='Zn', atomic_number=30, period=4, spin=0),
    Element(symbol='Ga', atomic_number=31, period=4),
    Element(symbol='Ge', atomic_number=32, period=4),
    Element(symbol='As', atomic_number=33, period=4),
    Element(symbol='Se', atomic_number=34, period=4),
    Element(symbol='Br', atomic_number=35, period=4),
    Element(symbol='Kr', atomic_number=36, period=4),
    Element(symbol='Rb', atomic_number=37, period=5),
    Element(symbol='Sr', atomic_number=38, period=5),
    Element(symbol='Y', atomic_number=39, period=5, spin=1),
    Element(symbol='Zr', atomic_number=40, period=5, spin=2),
    Element(symbol='Nb', atomic_number=41, period=5, spin=5),
    Element(symbol='Mo', atomic_number=42, period=5, spin=6),
    Element(symbol='Tc', atomic_number=43, period=5, spin=5),
    Element(symbol='Ru', atomic_number=44, period=5, spin=4),
    Element(symbol='Rh', atomic_number=45, period=5, spin=3),
    Element(symbol='Pd', atomic_number=46, period=5, spin=0),
    Element(symbol='Ag', atomic_number=47, period=5, spin=1),
    Element(symbol='Cd', atomic_number=48, period=5, spin=0),
    Element(symbol='In', atomic_number=49, period=5),
    Element(symbol='Sn', atomic_number=50, period=5),
    Element(symbol='Sb', atomic_number=51, period=5),
    Element(symbol='Te', atomic_number=52, period=5),
    Element(symbol='I', atomic_number=53, period=5),
    Element(symbol='Xe', atomic_number=54, period=5),
    Element(symbol='Cs', atomic_number=55, period=6),
    Element(symbol='Ba', atomic_number=56, period=6),
    Element(symbol='La', atomic_number=57, period=6),
    Element(symbol='Ce', atomic_number=58, period=6),
    Element(symbol='Pr', atomic_number=59, period=6),
    Element(symbol='Nd', atomic_number=60, period=6),
    Element(symbol='Pm', atomic_number=61, period=6),
    Element(symbol='Sm', atomic_number=62, period=6),
    Element(symbol='Eu', atomic_number=63, period=6),
    Element(symbol='Gd', atomic_number=64, period=6),
    Element(symbol='Tb', atomic_number=65, period=6),
    Element(symbol='Dy', atomic_number=66, period=6),
    Element(symbol='Ho', atomic_number=67, period=6),
    Element(symbol='Er', atomic_number=68, period=6),
    Element(symbol='Tm', atomic_number=69, period=6),
    Element(symbol='Yb', atomic_number=70, period=6),
    Element(symbol='Lu', atomic_number=71, period=6),
    Element(symbol='Hf', atomic_number=72, period=6),
    Element(symbol='Ta', atomic_number=73, period=6),
    Element(symbol='W', atomic_number=74, period=6),
    Element(symbol='Re', atomic_number=75, period=6),
    Element(symbol='Os', atomic_number=76, period=6),
    Element(symbol='Ir', atomic_number=77, period=6),
    Element(symbol='Pt', atomic_number=78, period=6),
    Element(symbol='Au', atomic_number=79, period=6),
    Element(symbol='Hg', atomic_number=80, period=6),
    Element(symbol='Tl', atomic_number=81, period=6),
    Element(symbol='Pb', atomic_number=82, period=6),
    Element(symbol='Bi', atomic_number=83, period=6),
    Element(symbol='Po', atomic_number=84, period=6),
    Element(symbol='At', atomic_number=85, period=6),
    Element(symbol='Rn', atomic_number=86, period=6),
    Element(symbol='Fr', atomic_number=87, period=7),
    Element(symbol='Ra', atomic_number=88, period=7),
    Element(symbol='Ac', atomic_number=89, period=7),
    Element(symbol='Th', atomic_number=90, period=7),
    Element(symbol='Pa', atomic_number=91, period=7),
    Element(symbol='U', atomic_number=92, period=7),
    Element(symbol='Np', atomic_number=93, period=7),
    Element(symbol='Pu', atomic_number=94, period=7),
    Element(symbol='Am', atomic_number=95, period=7),
    Element(symbol='Cm', atomic_number=96, period=7),
    Element(symbol='Bk', atomic_number=97, period=7),
    Element(symbol='Cf', atomic_number=98, period=7),
    Element(symbol='Es', atomic_number=99, period=7),
    Element(symbol='Fm', atomic_number=100, period=7),
    Element(symbol='Md', atomic_number=101, period=7),
    Element(symbol='No', atomic_number=102, period=7),
    Element(symbol='Lr', atomic_number=103, period=7),
    Element(symbol='Rf', atomic_number=104, period=7),
    Element(symbol='Db', atomic_number=105, period=7),
    Element(symbol='Sg', atomic_number=106, period=7),
    Element(symbol='Bh', atomic_number=107, period=7),
    Element(symbol='Hs', atomic_number=108, period=7),
    Element(symbol='Mt', atomic_number=109, period=7),
    Element(symbol='Ds', atomic_number=110, period=7),
    Element(symbol='Rg', atomic_number=111, period=7),
    Element(symbol='Cn', atomic_number=112, period=7),
    Element(symbol='Nh', atomic_number=113, period=7),
    Element(symbol='Fl', atomic_number=114, period=7),
    Element(symbol='Mc', atomic_number=115, period=7),
    Element(symbol='Lv', atomic_number=116, period=7),
    Element(symbol='Ts', atomic_number=117, period=7),
    Element(symbol='Og', atomic_number=118, period=7),
)

ELEMENT_BY_SYMBOL = {e.symbol: e for e in _ELEMENTS}
ELEMENT_BY_ATOMIC_NUM = {e.atomic_number: e for e in _ELEMENTS}

# Charge to valency, incomplete
VALENCY = {
    1: 1,
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    10: 0,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    15: 3,
    16: 2,
    17: 1,
    18: 0
}

# Offsets for indexing core orbitals
CORE_OFFSETS = {
    i: j
    for i, j in zip(
        np.arange(1, max(VALENCY.keys()) + 2),
        np.cumsum(np.concatenate([
            [0],
            np.fromiter(VALENCY.keys(), dtype=int) - np.fromiter(VALENCY.values(), dtype=int)
        ]))//2
    )
}

_keys = list(VALENCY.keys())
_total_off = 0
VAL_OFFSET = np.zeros((len(_keys)+1, len(_keys)+1), dtype=int)
for i in range(len(_keys)-1, -1, -1):
    for j in range(i, -1, -1):
        VAL_OFFSET[_keys[i], _keys[j]] = _total_off
        VAL_OFFSET[_keys[j], _keys[i]] = _total_off
        _total_off += min(VALENCY[_keys[i]], VALENCY[_keys[j]])

MAX_CORE_ORB = max(CORE_OFFSETS.values())
MAX_VAL_ORB = _total_off
# MAX_ORB = MAX_CORE_ORB + MAX_VAL_ORB
MAX_ORB = MAX_CORE_ORB + 4
