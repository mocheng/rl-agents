import copy
import importlib
import json
import logging
import gymnasium as gym

from rl_agents.configuration import Configurable

logger = logging.getLogger(__name__)

GYM_RENDER_MODES = {
    "headless": None,
    "human": "human",
    "video": "rgb_array",
}


def agent_factory(environment, config):
    """
        Handles creation of agents.

    :param environment: the environment
    :param config: configuration of the agent, must contain a '__class__' key
    :return: a new agent
    """
    if "__class__" in config:
        path = config['__class__'].split("'")[1]
        module_name, class_name = path.rsplit(".", 1)
        agent_class = getattr(importlib.import_module(module_name), class_name)
        agent = agent_class(environment, config)
        return agent
    else:
        raise ValueError("The configuration should specify the agent __class__")


def load_agent(agent_config, env):
    """
        Load an agent from a configuration file.

    :param agent_config: dict or the path to the agent configuration file
    :param env: the environment with which the agent interacts
    :return: the agent
    """
    # Load config from file
    if not isinstance(agent_config, dict):
        agent_config = load_agent_config(agent_config)
    return agent_factory(env, agent_config)


def load_agent_config(config_path):
    """
        Load an agent configuration from file, with inheritance.
    :param config_path: path to a json config file
    :return: the configuration dict
    """
    with open(config_path) as f:
        agent_config = json.loads(f.read())
    if "base_config" in agent_config:
        base_config = load_agent_config(agent_config["base_config"])
        del agent_config["base_config"]
        agent_config = Configurable.rec_update(base_config, agent_config)
    return agent_config


def load_environment(env_config, render_mode="video"):
    """
        Load an environment from a configuration file.

    :param env_config: the configuration, or path to the environment configuration file
    :param render_mode: output mode: headless, human, or video
    :return: the environment
    """
    # Load the environment config from file
    if not isinstance(env_config, dict):
        with open(env_config) as f:
            env_config = json.loads(f.read())

    # Make the environment
    if env_config.get("import_module", None):
        __import__(env_config["import_module"])
    if render_mode not in GYM_RENDER_MODES:
        raise ValueError("Unknown render mode '{}'.".format(render_mode))
    try:
        env = gym.make(env_config['id'], render_mode=GYM_RENDER_MODES[render_mode])
        # Save env module in order to be able to import it again
        env.import_module = env_config.get("import_module", None)
    except KeyError:
        raise ValueError("The gym register id of the environment must be provided")
    except gym.error.UnregisteredEnv:
        # The environment is unregistered.
        print("import_module", env_config["import_module"])
        raise gym.error.UnregisteredEnv('Environment {} not registered. The environment module should be specified by '
                                        'the "import_module" key of the environment configuration'.format(
                                            env_config['id']))

    # Configure the environment, if supported
    try:
        env.unwrapped.configure(env_config)
        # Reset the environment to ensure configuration is applied
        env.reset()
    except AttributeError as e:
        logger.info("This environment does not support configuration. {}".format(e))
    return env


def preprocess_env(env, preprocessor_configs, headless=True):
    """
        Apply a series of pre-processes to an environment, before it is used by an agent.

        Preprocessors typically return a copy of the environment, used for simulation or planning.
        With ``headless``, such copies are turned into headless shadow environments, so that
        simulating them never draws the planning process on the display.

    :param env: an environment
    :param preprocessor_configs: a list of preprocessor configs
    :param headless: whether the preprocessed copy should be a headless shadow environment
    :return: a preprocessed copy of the environment
    """
    original_env = env
    for preprocessor_config in preprocessor_configs:
        if "method" in preprocessor_config:
            try:
                preprocessor = getattr(env.unwrapped, preprocessor_config["method"])
                if "args" in preprocessor_config:
                    env = preprocessor(preprocessor_config["args"])
                else:
                    env = preprocessor()
            except AttributeError:
                logger.warning("The environment does not have a {} method".format(preprocessor_config["method"]))
        else:
            logger.error("The method is not specified in ", preprocessor_config)
    if headless and env is not original_env:
        disable_rendering(env)
    return env


def disable_rendering(env):
    """
        Turn an environment copy into a headless shadow environment, used for simulation only.

        Planning copies must not render: highway-env renders at every step when render_mode is
        'human', so a copied environment would create its own viewer and draw the simulated states
        on the display, showing the planning process in the user interface.

    :param env: an environment, or a copy of one
    """
    shadow = getattr(env, "unwrapped", env)
    try:
        shadow.viewer = None
        shadow.render_mode = None
    except AttributeError:
        pass


def safe_deepcopy_env(obj, headless=True):
    """
        Perform a deep copy of an environment but without copying its viewer.

        :param obj: the environment to deep-copy
        :param headless: if True (default), the copy is a headless shadow environment, used for
                         simulation and planning: it keeps no render mode and never renders on the
                         display. Set to False to preserve the render mode of the original env.
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    non_copied = ['viewer', '_monitor', 'grid_render', 'video_recorder', '_record_video_wrapper',
                  'screen', 'surf', 'clock', 'isopen']
    if headless:
        non_copied.append('render_mode')
    for k, v in obj.__dict__.items():
        if k not in non_copied:
            if isinstance(v, gym.Env):
                setattr(result, k, safe_deepcopy_env(v, headless=headless))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result
