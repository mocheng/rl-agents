"""
    JAX-native reinforcement learning agents and Jumanji environment bindings.

    This package is kept fully separate from `rl_agents`: legacy code never imports it,
    and it only depends on a small, stable subset of `rl_agents` (configuration, the
    AbstractAgent interface, the Transition namedtuple, the environment/agent factories,
    and the Evaluation trainer for subclassing).

    See JUMANJI_INTEGRATION.md for the design contract.
"""

__version__ = "0.1.0"
