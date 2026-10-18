def format_training_progress(name, env_steps, episodes, loss=None, epsilon=None, steps_per_second=None,
                            time=None):
    """
        Format one training progress line, shared by all training processes so that runs
        can be compared directly:

            [train DQNAgent] env_steps=5220 episodes=65 loss=nan eps=0.35 steps/s=129.4 time=42s

    :param name: the agent (or algorithm) name
    :param env_steps: cumulative number of environment steps
    :param episodes: cumulative number of completed episodes
    :param loss: latest training loss, if available
    :param epsilon: current exploration rate, if available
    :param steps_per_second: environment steps per second over the last interval
    :param time: cumulative wall-clock training time in seconds, if available
    :return: the formatted string
    """

    def fmt(value):
        return "{:.4g}".format(value) if value is not None else "nan"

    def fmt_time(value):
        return "{:.0f}s".format(value) if value is not None else "nan"

    return "[train {}] env_steps={} episodes={} loss={} eps={} steps/s={} time={}".format(
        name, env_steps, episodes, fmt(loss), fmt(epsilon), fmt(steps_per_second), fmt_time(time))


def log_training_progress(logger, name, env_steps, episodes, loss=None, epsilon=None,
                          steps_per_second=None, time=None):
    """
        Log a training progress line through the given logger.
    """
    logger.info(format_training_progress(name, env_steps, episodes, loss=loss, epsilon=epsilon,
                                         steps_per_second=steps_per_second, time=time))
