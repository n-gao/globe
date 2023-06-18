import functools

import flax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from chex import PRNGKey

from globe.nn.globe import Globe
from globe.nn.parameters import ParamTree
from globe.systems.scf import Scf
from globe.utils.config import SystemConfigs
from globe.utils.jax_utils import broadcast, instance, p_split, pmap, replicate
from globe.utils.optim import (NaturalGradientState,
                               make_natural_gradient_preconditioner,
                               make_schedule, scale_by_trust_ratio_embeddings)
from globe.vmc.hamiltonian import make_local_energy_function
from globe.vmc.mcmc import make_mcmc
from globe.vmc.optim import (make_loss_and_natural_gradient_fn,
                             make_std_based_damping_fn, make_training_step)
from globe.vmc.pretrain import (eval_orbitals, make_mol_param_loss,
                                make_pretrain_step)


class Trainer:
    """
    Trainer class for training the electronic wave function.
    Provides methods for training the wave function with VMC and for pretraining the molecular parameters.
    Also, provides methods for saving and loading the parameters.

    Args:
    - key: random key
    - mbnet: dictionary of parameters for the electronic wave function
    - mcmc_steps: number of MCMC steps
    - cg: dictionary of parameters for the conjugate gradient solver
    - loss: dictionary of parameters for the loss function
    - lr: dictionary of parameters for the learning rate schedule
    - damping: dictionary of parameters for the damping schedule
    """
    def __init__(self, key: PRNGKey, globe: dict, mcmc_steps: int, cg: dict, loss: dict, lr: dict, damping: dict):
        self.globe_params = globe
        self.cg_params = cg
        self.loss_params = loss
        self.lr_params = lr
        self.damping_params = damping
        self.mcmc_steps = mcmc_steps
        # Prepare random keys
        self.key, *subkeys = jax.random.split(key, jax.device_count()+1)
        self.shared_key = broadcast(jnp.stack(subkeys))


        # Prepare all necessary functions
        self.network = Globe(**globe)
        self.wf = functools.partial(self.network.apply, method=self.network.wf)
        self.p_wf = pmap(jax.vmap(self.wf, in_axes=(None, 0, None, None, None)), in_axes=(0, 0, None, None, 0), static_broadcasted_argnums=3)
        self.fwd = pmap(jax.vmap(self.network.apply, in_axes=(None, 0, None, None)), in_axes=(0, 0, None, None), static_broadcasted_argnums=3)
        self.get_mol_params = functools.partial(self.network.apply, method=self.network.get_mol_params)
        self.get_mol_params = jax.jit(self.get_mol_params, static_argnums=2)
        self.p_get_mol_params = pmap(self.get_mol_params, in_axes=(0, None, None), static_broadcasted_argnums=2)
        self.get_intermediates = jax.vmap(functools.partial(self.network.apply, capture_intermediates=True), in_axes=(None, 0, None, None))
        self.p_get_intermediates = pmap(self.get_intermediates, in_axes=(0, 0, None, None), static_broadcasted_argnums=3)

        # Initialize parameters
        self.init_params(jnp.ones((2, 3)), jnp.zeros((1, 3)), SystemConfigs(((1, 1),), ((2,),)))

        # Prepare VMC training functions
        self.wf_mcmc = make_mcmc(self.wf, self.mcmc_steps)
        self.p_wf_mcmc = pmap(self.wf_mcmc, in_axes=(0, 0, None, None, 0, 0, None), static_broadcasted_argnums=3)
        @functools.partial(jax.jit, static_argnums=3)
        def mcmc(params, electrons, atoms, config, key, widths):
            mol_params = self.get_mol_params(params, atoms, config)
            return self.wf_mcmc(params, electrons, atoms, config, mol_params, key, widths)
        self.mcmc = mcmc

        pre_mcmc = make_mcmc(self.wf, 1)
        p_pre_mcmc = pmap(pre_mcmc, in_axes=(0, 0, None, None, 0, 0, None), static_broadcasted_argnums=3)
        @functools.partial(jax.jit, static_argnums=3)
        def p_mcmc(params, electrons, atoms, config, key, widths):
            mol_params = self.get_mol_params(params, atoms, config)
            return pre_mcmc(params, electrons, atoms, config, mol_params, key, widths)
        

        self.wf_energy = make_local_energy_function(self.wf, self.network.group_parameters, linearize=self.loss_params['linearize'])
        @functools.partial(jax.jit, static_argnums=3)
        def energy_fn(params, electrons, atoms, config):
            mol_params = self.get_mol_params(params, atoms, config)
            return self.wf_energy(params, electrons, atoms, config, mol_params)
        self.energy = energy_fn
        self.v_energy = jax.vmap(self.energy, in_axes=(None, 0, None, None))
        self.p_energy = pmap(self.v_energy, in_axes=(0, 0, None, None), static_broadcasted_argnums=3)

        self.natgrad_precond = make_natural_gradient_preconditioner(
            self.network,
            **self.cg_params
        )

        self.loss_and_grad = make_loss_and_natural_gradient_fn(
            self.network.apply,
            natgrad_precond=self.natgrad_precond,
            **self.loss_params,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_schedule(make_schedule(lr)),
            optax.scale(-1.0)
        )

        self.damping_fn = make_std_based_damping_fn(**self.damping_params)
        self.train_step = make_training_step(
            self.mcmc,
            self.loss_and_grad,
            self.energy,
            self.damping_fn,
            self.optimizer.update
        )

        # Prepare pretraining functions
        kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: 'kernel' in p)
        embeddings = flax.traverse_util.ModelParamTraversal(lambda p, _: 'embedding' in p)
        prefixed_params = flax.traverse_util.ModelParamTraversal(lambda p, _: 'prefixed' in p)

        all_false = jtu.tree_map(lambda _: False, self.params['params'])
        kernel_mask = kernels.update(lambda _: True, all_false)
        embedding_mask = embeddings.update(lambda _: True, all_false)
        prefixed_mask = prefixed_params.update(lambda _: True, all_false)

        self.pre_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.masked(optax.set_to_zero(), prefixed_mask),
            optax.scale_by_adam(),
            optax.masked(optax.scale_by_trust_ratio(), kernel_mask),
            optax.masked(scale_by_trust_ratio_embeddings(), embedding_mask),
            optax.scale_by_schedule(lambda t: -1e-3 * 1/(1+t/1000)),
        )
        
        self.pre_step = make_pretrain_step(
            p_mcmc,
            self.get_mol_params,
            functools.partial(self.network.apply, function='orbitals', method=self.network.wf),
            self.pre_optimizer.update,
            self.network.full_det,
            mol_param_aux_loss=make_mol_param_loss(self.network.param_spec(), 1e-6 if self.network.meta_model != 'none' else 0)
        )

        # Init data
        self._opt_state = None
        self._pre_opt_state = None
        self.iteration = 0
        self.init_natgrad_state()
    
    def init_params(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs):
        """
        Initialize the parameters of the network.
        
        Args:
        - electrons: The electrons in the system.
        - atoms: The atoms in the system.
        - config: The system configuration.
        """
        self.key, subkey = jax.random.split(self.key)
        self.params = self.network.init(
            subkey,
            electrons,
            atoms,
            config
        ).unfreeze()
        self.params = replicate(self.params)
    
    def init_natgrad_state(self):
        """
        Initialize the natural gradient state.
        """
        self.natgrad_state = NaturalGradientState(
            damping=replicate(jnp.array(self.damping_params['init'], dtype=jnp.float32)),
            last_grad=broadcast(jtu.tree_map(jnp.zeros_like, self.params['params']))
        )

    def mol_params(self, atoms: jax.Array, config: SystemConfigs):
        """
        Get the adaptive parameters for the specified systems.
        
        Args:
        - atoms: The atoms in the system.
        - config: The system configurations.
        """
        return self.get_mol_params(instance(self.params), atoms, config)

    @property
    def opt_state(self):
        if self._opt_state is None:
            self._opt_state = pmap(self.optimizer.init)(self.params['params'])
            self.iteration = 0
            self.init_natgrad_state()
        return self._opt_state
    
    @opt_state.setter
    def opt_state(self, val):
        self._opt_state = val
    
    @property
    def pre_opt_state(self):
        if self._pre_opt_state is None:
            self._pre_opt_state = pmap(self.pre_optimizer.init)(self.params['params'])
            self.init_natgrad_state()
        return self._pre_opt_state
    
    @pre_opt_state.setter
    def pre_opt_state(self, val):
        self._pre_opt_state = val
    
    def next_key(self):
        self.shared_key, result = p_split(self.shared_key)
        return result
    
    def intermediates(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs) -> ParamTree:
        """
        Get all intermediate tensors of the network's forward pass.
        
        Args:
        - electrons: The electrons in the system.
        - atoms: The atoms in the system.
        - config: The system configuration.
        Returns:
        - intermediates: The intermediate tensors.
        """
        return self.p_get_intermediates(self.params, electrons[:1, :1], atoms, config)[1]

    def pretrain_step(
            self,
            electrons: jax.Array,
            atoms: jax.Array,
            config: SystemConfigs,
            scfs: list[Scf],
            properties: dict[str, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Perform a single pretraining step.
        
        Args:
        - electrons: The electrons in the systems.
        - atoms: The atoms in the systems.
        - config: The system configurations.
        - scfs: The SCF objects for the systems.
        - properties: The properties of the systems.
        Returns:
        - loss: The loss of the pretraining step.
        - electrons: The electrons in the systems.
        - pmove: The proportion of electrons that moved.
        """
        targets = eval_orbitals(scfs, electrons, config.spins)
        self.params, electrons, self.pre_opt_state, self.natgrad_state, loss, pmove = self.pre_step(
            self.params,
            electrons,
            atoms,
            config,
            targets,
            self.pre_opt_state,
            self.next_key(),
            properties,
            self.natgrad_state
        )
        return loss, electrons, pmove

    def step(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs, properties: dict[str, jax.Array]) -> tuple[jax.Array, dict, dict]:
        """
        Perform a single training step.
        
        Args:
        - electrons: The electrons in the systems.
        - atoms: The atoms in the systems.
        - config: The system configurations.
        - properties: The properties of the systems.
        Returns:
        - electrons: The electrons in the systems.
        - mol_data: Data per molecular structure.
        - aux_data: The auxiliary data to log.
        """
        (electrons, self.params, self.opt_state, self.natgrad_state), mol_data, aux_data = self.train_step(
            self.params,
            electrons,
            atoms,
            config,
            self.opt_state,
            self.next_key(),
            self.natgrad_state,
            properties
        )
        self.iteration += 1
        return electrons, mol_data, aux_data
    
    def serialize_params(self) -> bytes:
        """
        Serialize the parameters of the network.
        """
        to_store = instance(self.params)
        return flax.serialization.msgpack_serialize(to_store)

    def load_params(self, blob: bytes):
        """
        Load the parameters of the network.
        
        Args:
        - blob: The serialized parameters.
        """
        data = flax.serialization.msgpack_restore(blob)
        self.params = replicate(data)
