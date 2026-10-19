import re

import logging

from rl_agents.trainer.evaluation import Evaluation

logger = logging.getLogger(__name__)


def save_rotating_checkpoint(evaluation, identifier, keep=3):
    """
        Save a run checkpoint and keep only the last `keep` checkpoints in the run directory,
        deleting older ones. `saved_models/latest.tar` is unaffected (it is overwritten by
        `Evaluation.save_agent_model`).

    :param evaluation: the Evaluation instance
    :param identifier: the checkpoint identifier (e.g. the batch index)
    :param keep: number of most recent `checkpoint-<identifier>.tar` files to keep
    """
    evaluation.save_agent_model(identifier)
    pattern = re.compile(r"checkpoint-(\d+)\.tar")
    checkpoints = []
    for path in evaluation.run_directory.glob("checkpoint-*.tar"):
        match = pattern.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    for _, path in sorted(checkpoints)[:-keep]:
        path.unlink()
        logger.info("Removed old checkpoint {}".format(path.name))


class TorchRLEvaluation(Evaluation):
    """
        The evaluation of a torchrl agent: training is delegated to the agent's batch-collection
        loop (`TorchRLDQNAgent.train_loop`), while run metadata, TensorBoard summaries and
        checkpointing are inherited from `Evaluation`.
    """

    def train(self):
        self.training = True
        try:
            self.agent.seed(self.sim_seed)
            self.agent.reset()
            self.agent.train_loop(self)
        finally:
            self.close()

    def close(self):
        try:
            self.agent.close()
        except AttributeError:
            pass
        super().close()
