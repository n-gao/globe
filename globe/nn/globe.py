from typing import Type
import flax.linen as nn

import globe.nn.orbitals as ORBITALS
from globe.nn.ferminet import FermiNet
from globe.nn.orbitals import OrbitalModule, get_orbitals
from globe.nn.parameters import ParamSpec, ParamType, group_parameters
from globe.nn.moon import Moon
from globe.nn.metanet import MetaNet, PlaceHolderGNN
from globe.nn.coords import find_axes, find_atom_frames


_WAVE_FUNCTIONS = {
    'ferminet': FermiNet,
    'moon': Moon
}

_GNNS = {
    'metanet': MetaNet,
    'none': PlaceHolderGNN
}


class Globe(nn.Module):
    wf_params: dict
    gnn_params: dict
    shared_orbitals: bool
    orbital_type: str | Type[OrbitalModule]
    orbital_config: dict
    determinants: int = 16
    full_det: bool = True
    wf_model: str = 'moon'
    meta_model: str = 'metanet'

    def setup(self):
        assert self.wf_model in _WAVE_FUNCTIONS
        assert self.meta_model in _GNNS
        if isinstance(self.orbital_type, str):
            orbital_type = getattr(ORBITALS, self.orbital_type)
        else:
            orbital_type = self.orbital_type
        
        wf_cls = _WAVE_FUNCTIONS[self.wf_model]
        self.wave_function = wf_cls(
            orbital_type=orbital_type,
            orbital_config=self.orbital_config,
            **{
                **self.wf_params['shared'],
                **self.wf_params[self.wf_model]
            }
        )
        param_spec = self.param_spec()
        # delete axes from spec as we will overwrite it
        del param_spec['axes']
        self.gnn = _GNNS[self.meta_model].create(param_spec, **self.gnn_params)
    
    def param_spec(self):
        if isinstance(self.orbital_type, str):
            orbital_type = getattr(ORBITALS, self.orbital_type)
        else:
            orbital_type = self.orbital_type
        param_spec = _WAVE_FUNCTIONS[self.wf_model].spec(
            shared_orbitals=self.shared_orbitals,
            full_det=self.full_det,
            determinants=self.determinants,
            orbital_type=orbital_type,
            orbital_config=self.orbital_config,
            **self.wf_params[self.wf_model]
        )
        # We add the axes to ensure proper grouping
        param_spec['axes'] = ParamSpec(ParamType.GLOBAL, (3, 3), 0, 0)
        param_spec['atom_frames'] = ParamSpec(ParamType.NUCLEI, (3, 3), 0, 0)
        return param_spec
    
    def group_parameters(self, mol_params, config, groups=None):
        return group_parameters(mol_params, self.param_spec(), config, groups=groups)

    def structure_params(self, atoms, config):
        return {
            'axes': find_axes(atoms, config),
            'orbitals': get_orbitals(atoms, config),
            'atom_frames': find_atom_frames(atoms, config)
        }

    def get_mol_params(self, atoms, config, struc_params=None):
        if struc_params is None:
            struc_params = self.structure_params(atoms, config)
        orbital_params = self.gnn(atoms, struc_params, config).unfreeze()
        orbital_params['axes'] = struc_params['axes']
        orbital_params['atom_frames'] = struc_params['atom_frames']
        return orbital_params
    
    def orbitals(self, electrons, atoms, config, struc_params=None):
        mol_params = self.get_mol_params(atoms, config, struc_params)
        return self.wave_function.orbitals(electrons, atoms, config, mol_params)
    
    def wf(self, electrons, atoms, config, mol_params, function='__call__'):
        return getattr(self.wave_function, function)(electrons, atoms, config, mol_params)
    
    def __call__(self, electrons, atoms, config, struc_params=None):
        mol_params = self.get_mol_params(atoms, config, struc_params)
        return self.wave_function(electrons, atoms, config, mol_params)
