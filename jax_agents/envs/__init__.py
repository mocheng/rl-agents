"""
    Gymnasium registration of Jumanji environments through the `JumanjiEnv` wrapper.

    Importing this module registers environment ids such as `jumanji/Sokoban-v0`, so
    that environment configuration files can follow the usual repository pattern:

        {
            "id": "jumanji/Sokoban-v0",
            "import_module": "jax_agents.envs",
            "make_kwargs": {
                "generator": {"__class__": "jumanji.environments.routing.sokoban.generator.ToyGenerator"},
                "time_limit": 120
            }
        }

    Note: importing this module does not import JAX/Jumanji; those imports happen lazily
    when an environment is actually instantiated through `gym.make`.
"""

from gymnasium.envs.registration import register

register(
    id="jumanji/Sokoban-v0",
    entry_point="jax_agents.envs.jumanji:JumanjiEnv",
    kwargs={"env_id": "Sokoban-v0"},
)
