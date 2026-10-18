import logging

from rl_agents.trainer.evaluation import Evaluation

logger = logging.getLogger(__name__)


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
