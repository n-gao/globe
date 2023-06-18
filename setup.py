from setuptools import find_packages, setup

install_requires = [
    'numpy',
    'scipy',
    'tqdm',
    'jax>=0.4, <0.5',
    'flax',
    'jraph',
    'pyscf',
    'optax',
    'seml',
    'seml_logger @ git+https://github.com/n-gao/seml_logger.git@927bba7e47de45deb9a53e472159199dd31b7b0e#egg=seml_logger',
    'sacred'
]


setup(name='globe',
      version='0.1.0',
      description='Graph-learned orbital embeddings & Molecular orbital network',
      packages=find_packages('.'),
      install_requires=install_requires,
      zip_safe=False)
