import functools
import logging
from collections import Counter, defaultdict
from typing import Tuple

import jax
import jax.tree_util as jtu
import numpy as np
import seml
from sacred import Experiment
from seml_logger import Logger, add_default_observer_config, automain

import globe.systems as Systems
import globe.systems.property as Properties
from globe.systems.dataset import Dataset
from globe.trainer import Trainer

jax.config.update('jax_default_matmul_precision', 'float32')

ex = Experiment()
seml.setup_logger(ex)
add_default_observer_config(ex, notify_on_completed=True)


@ex.config
def config():
    globe = dict(
        wf_params=dict(
            ferminet=dict(
                hidden_dims=((256, 32), (256, 32), (256, 32), (256, 32)),
                embedding_dim=256,
                embedding_adaptive_weights=True,
                restricted=True
            ),
            moon=dict(
                hidden_dims=(256,),
                use_interaction=False,
                update_before_int=4,
                update_after_int=0,
                adaptive_update_bias=True,
                local_frames=False,

                edge_embedding='MLPEdgeEmbedding',
                edge_embedding_params=dict(
                    # MLPEdgeEmbedding
                    out_dim=8,
                    hidden_dim=16,
                    activation='silu',
                    adaptive_weights=True,
                    init_sph=False,
                ),

                embedding_dim=256,
                embedding_e_out_dim=256,
                embedding_int_dim=32,
                embedding_adaptive_weights=True,
                embedding_adaptive_norm=False,
            ),
            attentive=dict(
                head_dim=64,
                heads=4,
                layer=4,
                use_layernorm=True,
                include_spin_emb=True,
                # Override defaults
                activation='tanh',
                jastrow_mlp_layers=0,
            ),
            shared=dict(
                activation='silu',
                jastrow_mlp_layers=3,
                jastrow_include_pair=True,
                adaptive_sum_weights=False,
                adaptive_jastrow=False,
            )
        ),
        gnn_params=dict(
            layers=((64, 128), (64, 128), (64, 128)),
            embedding_dim=128,
            edge_embedding='MLPEdgeEmbedding',
            edge_embedding_params=dict(
                # MLPEdgeEmbedding
                out_dim=16,
                hidden_dim=64,
                activation='silu',
                adaptive_weights=True,
                adaptive=False,
                init_sph=False,
                # SphHarmEmbedding
                n_rad=6,
                max_l=3,
            ),
            orb_edge_params=dict(
                param_type='orbital'
            ),
            out_mlp_depth=3,
            out_scale='log',
            aggregate_before_out=True,
            activation='silu',
            enable_groups=False,
            enable_segments=False
        ),
        orbital_type='ProductOrbitals',
        orbital_config=dict(
            separate_k=True
        ),
        determinants=16,
        full_det=True,
        shared_orbitals=True,
        meta_model='metanet',
        wf_model='moon'
    )
    mcmc_steps = 40
    cg = dict(
        maxiter=100,
        precision='float32',
        linearize=True
    )
    loss = dict(
        clip_local_energy=5.0,
        limit_scaling=True,
        target_std=1.0,
        linearize=True
    )
    lr = dict(
        init=0.1,
        delay=100,
        decay=1
    )
    damping = dict(
        init=1e-3,
        base=1e-4
    )

    batch_size = 16
    batch_behavior = 'fill_random'
    samples_per_batch = 4096
    thermalizing_steps = 1000
    chkpts = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

    properties = (
        ('WidthScheduler', dict(
            init_width=0.02,
            target_pmove=0.525,
            update_interval=20
        )),
        ('EnergyStdEMA', dict(
            decay=0.99
        )),
        ('EnergyEMA', dict(
            decay=0.99
        ))
    )

    restricted = True
    basis = 'STO-6G'

    pretrain_epochs = 10000
    training_epochs = 10000
    
    pretrain_systems = None

    chkpt = None


def naming_fn(systems):
    molecules = Systems.get_molecules(systems)
    return '-'.join([f'{k}_{c}' for k, c in Counter(molecules).items()])


def themalize_dataset(trainer, dataset, logger, steps):
    logging.info('Thermalizing')
    for batch in dataset:
        electrons, atoms, config, props = batch.to_jax()
        mol_params = trainer.p_get_mol_params(trainer.params, atoms, config)
        for _ in logger.tqdm(range(steps)):
            electrons, atoms, config, props = batch.to_jax()
            electrons, pmove = trainer.p_wf_mcmc(
                trainer.params,
                electrons,
                atoms,
                config,
                mol_params,
                trainer.next_key(),
                props['mcmc_width']
            )
            batch.update_states(electrons, pmove=pmove)
            logger.set_postfix({
                'pmove': np.mean(pmove).item()
            })


@automain(ex, naming_fn, default_folder='~/logs/dev_globe')
def main(
    seed: int,

    pretrain_systems: dict,
    systems: dict,

    globe: dict,
    mcmc_steps: int,
    cg: dict,
    loss: dict,
    lr: dict,
    damping: dict,

    batch_size: int,
    batch_behavior: str,
    samples_per_batch: int,
    thermalizing_steps: int,
    chkpts: Tuple[int, ...],

    properties: Tuple[Tuple[str, dict], ...],

    restricted: bool,
    basis: str,

    pretrain_epochs: int,
    training_epochs: int,
    
    chkpt: str,

    logger: Logger = None,
):
    chkpts = set(chkpts)
    key = jax.random.PRNGKey(seed)
    n_devices = jax.device_count()
    key, subkey = jax.random.split(key)
    trainer = Trainer(subkey, globe, mcmc_steps, cg, loss, lr, damping)

    # Initialize pretraining dataset
    key, subkey = jax.random.split(key)
    mols = Systems.get_molecules(pretrain_systems if pretrain_systems is not None else systems)
    for m in set(mols):
        logger.add_tag(str(m))
    # if we have fewer molecules update this variable accordingly
    eff_batch_size = min((batch_size, len(mols)))
    # we divide and multiple by n_devices to ensure that the batches can be parallized across multiple GPUs.
    samples_per_molecule = samples_per_batch//eff_batch_size//n_devices * n_devices
    dataset = Dataset(
        subkey, 
        mols,
        'random', 
        batch_behavior,
        eff_batch_size,
        samples_per_molecule,
        (functools.partial(Properties.WidthScheduler, init_width=0.02),),
        restricted,
        basis
    )

    # Initialization
    if chkpt is not None:
        logging.info(f'Loading checkpoint: {chkpt}')
        with open(chkpt, 'rb') as inp:
            trainer.load_params(inp.read())

        # Thermalize
        if pretrain_epochs > 0:
            themalize_dataset(trainer, dataset, logger, thermalizing_steps)
    else:
        electrons, atoms, config, _ = next(iter(dataset)).to_jax()
        trainer.init_params(electrons[0, 0], atoms, config)
    
    
    # Pretrain
    logging.info('Pretraining')
    i = 0
    for epoch in logger.tqdm(range(pretrain_epochs)):
        for batch in dataset:
            electrons, atoms, config, props = batch.to_jax()
            loss, electrons, pmove = trainer.pretrain_step(electrons, atoms, config, batch.scfs, props)
            batch.update_states(electrons, pmove=pmove)
            if np.isnan(loss).any():
                raise RuntimeError(f"Encountered NaNs in pretraining step {i}!")
            logger.add_scalar('pertrain/loss', loss.mean().item(), step=i)
            logger.add_scalar('pertrain/pmove', pmove.mean().item(), step=i)
            i += 1
            logger.set_postfix({
                'loss': loss.mean().item()
            })
        # Log parameters
        if epoch % 1000 == 0 or epoch == pretrain_epochs-1:
            with logger.without_aim():
                logger.add_distribution_dict(trainer.params, 'pretrain', step=epoch)
                logger.add_distribution_dict(trainer.mol_params(atoms, config), 'pretrain/mol_params', step=epoch)
    
    # Initialize VMC dataset
    key, subkey = jax.random.split(key)
    mols = Systems.get_molecules(systems)
    # if we have fewer molecules update this variable accordingly
    eff_batch_size = min((batch_size, len(mols)))
    # we divide and multiple by n_devices to ensure that the batches can be parallized across multiple GPUs.
    samples_per_molecule = samples_per_batch//eff_batch_size//n_devices * n_devices
    dataset = Dataset(
        subkey, 
        mols, 
        'random', 
        batch_behavior,
        eff_batch_size,
        samples_per_molecule,
        tuple(
            functools.partial(getattr(Properties, prop_name), **kwargs)
            for prop_name, kwargs in properties
        ),
        restricted,
        basis
    )

    # Thermalize
    themalize_dataset(trainer, dataset, logger, thermalizing_steps)

    # VMC training
    logging.info('VMC Training')
    i = 0
    for epoch in logger.tqdm(range(training_epochs)):
        epoch_data = defaultdict(list)
        for batch in dataset:
            electrons, atoms, config, props = batch.to_jax()
            electrons, mol_data, aux_data = trainer.step(electrons, atoms, config, props)
            batch.update_states(electrons, **mol_data)
            # Move to CPU and reduce parallel GPU dimension
            aux_data = jtu.tree_map(lambda x: np.mean(x, 0), aux_data)
            if np.isnan(aux_data['E']).any():
                raise RuntimeError(f"Encountered NaNs in step {i}!")
            logger.add_scalar_dict(jtu.tree_map(lambda x: np.mean(x), aux_data), 'train', step=i)
            for mol, e, e_var in zip(batch.molecules, aux_data['E'], aux_data['E_var']):
                epoch_data[mol].append((e, e_var))
            i += 1
        # Log per molecule data
        postfix = {'E': {}, 'E_std': {}}
        for m, data in epoch_data.items():
            data = np.array(data)
            E = data[:, 0].mean()
            E_std = data[:, 1].mean() ** 0.5
            postfix['E'][str(m)] = E
            postfix['E_std'][str(m)] = E_std
            logger.add_scalar('mol/E', E, step=i, context={'subset': f'{m}'})
            logger.add_scalar('mol/E_std', E_std, step=i, context={'subset': f'{m}'})
        logger.set_postfix(postfix)
        # Log parameters
        if (epoch < 1000 and epoch % 100 == 0) or epoch % 1000 == 0:
            with logger.without_aim():
                logger.add_distribution_dict(trainer.params, step=epoch, context={'subset': 'train'})
                logger.add_distribution_dict(trainer.mol_params(atoms, config), 'mol_params', step=epoch, context={'subset': 'train'})
                # logger.add_distribution_dict(trainer.inteErmediates(electrons, atoms, config), step=epoch, context={'subset': 'train'})
        if epoch % 10000 == 0 or epoch in chkpts:
            logging.info(f'Checkpoint {epoch}')
            logger.store_blob(f'chk_{epoch}.chk', trainer.serialize_params())
            logger.store_data(f'chk_{epoch}', {
                repr(mol): jtu.tree_map(lambda x: np.array(x) if isinstance(x, jax.Array) else x, mol.property_values)
                for mol in dataset.molecules
            }, use_json=True, use_pickle=False)
    logger.store_blob('chk_final.chk', trainer.serialize_params())
    training_results = {
        repr(mol): jtu.tree_map(lambda x: np.array(x) if isinstance(x, jax.Array) else x, mol.property_values)
        for mol in dataset.molecules
    }

    return {
        'training': training_results
    }
