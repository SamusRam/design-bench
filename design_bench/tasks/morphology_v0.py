from design_bench import DATA_DIR
from design_bench import maybe_download
from design_bench.task import Task
from morphing_agents.mujoco.ant.env import MorphingAntEnv
from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.ant.elements import LEG
from morphing_agents.mujoco.ant.elements import LEG_LOWER_BOUND
from morphing_agents.mujoco.ant.elements import LEG_UPPER_BOUND
import numpy as np
import os
import pickle as pkl


class MorphologyV0Task(Task):

    def score(self, x):
        return NotImplemented

    def __init__(self,
                 env_class=MorphingAntEnv,
                 elements=4,
                 env_element=LEG,
                 env_element_lb=LEG_LOWER_BOUND,
                 env_element_ub=LEG_UPPER_BOUND,
                 oracle_weights='ant_oracle.pkl',
                 x_file='ant_morphology_X.npy',
                 y_file='ant_morphology_y.npy',
                 split_percentile=80):
        """Load static datasets of weights and their corresponding
        expected returns from the disk

        Args:

        num_parallel: int
            the number of parallel trials in the docker container
        num_gpus: int
            the number of gpus to use in this docker container
        n_envs: int
            the number of parallel sampling environments for PPO
        max_episode_steps: int
            the maximum length of a episode when collecting samples
        total_timesteps: int
            the maximum number of samples to collect from the environment
        domain: str
            the particular morphology domain such as 'ant' or 'dog'
        """

        maybe_download('12H-4AvzpMVmq7M7b7nD_RPu5GNeCuCbu',
                       os.path.join(DATA_DIR, 'ant_morphology_X.npy'))
        maybe_download('1uSF6oc7OlLGioe_sZQwmjibuRbPCYRGu',
                       os.path.join(DATA_DIR, 'ant_morphology_y.npy'))
        maybe_download('1FJY6LQG3kvLIx00WT2XVGyfU9JrmO0T0',
                       os.path.join(DATA_DIR, 'ant_oracle.pkl'))
        maybe_download('17PqWmmbIiUwNBxLwhjTnV38Dx2ejefCQ',
                       os.path.join(DATA_DIR, 'dkitty_morphology_X.npy'))
        maybe_download('1DggTSnRn_SzmYonbLgLppzWC8OtqYAb8',
                       os.path.join(DATA_DIR, 'dkitty_morphology_y.npy'))
        maybe_download('1GpZc9UezjyvAn_ejwG3cibq1stQMjiiM',
                       os.path.join(DATA_DIR, 'dkitty_oracle.pkl'))

        self.env_class = env_class
        self.elements = elements
        self.env_element = env_element
        self.env_element_lb = env_element_lb
        self.env_element_ub = env_element_ub
        with open(os.path.join(
                DATA_DIR, oracle_weights), 'rb') as f:
            self.weights = pkl.load(f)

        x = np.load(os.path.join(DATA_DIR, x_file))
        y = np.load(os.path.join(DATA_DIR, y_file))
        x = x.astype(np.float32)
        y = y.astype(np.float32).reshape([-1, 1])

        # remove all samples above the qth percentile in the data set
        split_temp = np.percentile(y[:, 0], split_percentile)
        indices = np.where(y < split_temp)[0]
        self.x = x[indices].astype(np.float32)
        self.y = y[indices].astype(np.float32)
        self.score = np.vectorize(self.scalar_score, signature='(n)->(1)')

    def scalar_score(self,
                     x: np.ndarray) -> np.ndarray:
        """Calculates a score for the provided tensor x using a ground truth
        oracle function (the goal of the task is to maximize this)

        Args:

        x: np.ndarray
            a batch of sampled designs that will be evaluated by
            an oracle score function

        Returns:

        scores: np.ndarray
            a batch of scores that correspond to the x values provided
            in the function argument
        """

        # create a policy forward pass in numpy
        def mlp_policy(h):
            h = np.maximum(0.0, h @ self.weights[0] + self.weights[1])
            h = np.maximum(0.0, h @ self.weights[2] + self.weights[3])
            h = h @ self.weights[4] + self.weights[5]
            return np.tanh(np.split(h, 2)[0])

        # split x into a morphology supported by the environment
        design = [self.env_element(
            *np.clip(np.array(xi), self.env_element_lb, self.env_element_ub))
            for xi in np.split(x, self.elements)]

        # build an agent with this morphology in sim
        env = self.env_class(expose_design=False, fixed_design=design)
        obs = env.reset()
        ret = np.zeros([1], dtype=np.float32)

        # perform a single roll out for quick evaluation
        for i in range(1000):
            obs, rew, done, info = env.step(mlp_policy(obs))
            ret += rew.astype(np.float32)
            if done:
                break
        return ret
