from setuptools import setup, find_packages

setup(
    name='rl-agents',
    version='1.0.dev0',
    description='A collection of Reinforcement Learning agents',
    url='https://github.com/mocheng/rl-agents',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='reinforcement learning agents',
    packages=find_packages(exclude=['docs', 'scripts', 'tests*']),
    install_requires=['gymnasium', 'numpy', 'pandas', 'numba', 'pygame-ce', 'matplotlib', 'seaborn', 'six', 'docopt',
                      'torch>=1.2.0', 'tensorboardX', 'tensorboard', 'moviepy', 'scipy'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
        # JAX-native agents and Jumanji environments (see JUMANJI_INTEGRATION.md).
        # Note: the base install does not require JAX. Tested combination on macOS
        # arm64 / Python 3.13: jax==0.4.34, jumanji==1.1.2, flax==0.9.0, optax==0.2.4,
        # chex==0.1.89 (see the design doc for the jax-metal caveat).
        'jax': ['jax>=0.4.27,<0.8', 'jaxlib>=0.4.27,<0.8', 'jumanji>=1.1',
                'flax', 'optax', 'chex'],
    },
    entry_points={
        'console_scripts': [],
    },
)

