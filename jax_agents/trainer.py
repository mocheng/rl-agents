import logging

from rl_agents.trainer.evaluation import Evaluation

logger = logging.getLogger(__name__)


class JaxEvaluation(Evaluation):
    """
        The evaluation of a JAX agent interacting with a Jumanji environment.

        Training is delegated to the agent (`jax_agents.agents.base.JaxAgent.train_jax`),
        which runs jitted/vectorized rollouts and updates, reporting through this object.
        Everything else — run metadata, logging, TensorBoard summaries, best-model
        tracking and checkpointing (`--recover`) — is inherited from `Evaluation`.
    """

    def train(self):
        self.training = True
        try:
            self.agent.seed(self.sim_seed)
            self.agent.reset()
            self.agent.train_jax(self)
        finally:
            self.close()
