"""
Based on Stable Baselines 3 example:

    For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
    For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

    Authored by: Elliot (https://github.com/elliottower)
    Access: https://pettingzoo.farama.org/_modules/pettingzoo/classic/connect_four/connect_four/#env

# modification made by moroviintaas (https://github.com/moroviintaas) for purpose of creating baseline for rust implementation
"""



import glob
import os
import time

import gymnasium as gym
from pettingzoo.utils import env_logger, wrappers
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
from pettingzoo.classic import connect_four_v3


import argparse

from torch import nn



#from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description = "SB3 PPO (Selfplay) model for ConnectFour Game")
    parser.add_argument('-e', "--epochs", type=int, default=100, help="Number of epochs of training")
    parser.add_argument('-n', "--n-epochs", type=int, default=4, help="Number of update epochs per rollout")
    parser.add_argument('-g', "--games", type=int, default=128, help="Number of games in epochs of training")
    parser.add_argument("--limit-steps-in-epoch", type=int, default=1024, help = "Limit number of all steps in all epoch")
    parser.add_argument('-i', "--illegal-reward", type=float, default=-10, help="NPenalty for illegal actions")
    parser.add_argument('-p', "--penalty", type=float, default=-10, help="NPenalty for illegal actions")
    parser.add_argument("--learning-rate", dest="learning_rate", default = 1e-4, help = "Adam learning rate")
    parser.add_argument("--masking", action="store_true", dest="masking",  help = "Enable illegal action masking")
    parser.add_argument("--layer-sizes-0", metavar="LAYERS", type=int, nargs="*", default=[64,64], help = "Sizes of subsequent linear layers")
    parser.add_argument("--cuda",  action="store_true", help="enable cuda gpu")
    parser.add_argument('-m', "--minibatch-size", type=int, default=16, help="Size of PPO minibatch")
    return parser.parse_args()

# To pass into other gymnasium wrappers, we need to ensure that pettingzoo's wrappper
# can also be a gymnasium Env. Thus, we subclass under gym.Env as well.
class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info.

        The observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).
        """
        current_agent = self.agent_selection

        super().step(action)

        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]




def create_model(env_fn, config,  verbose=False, **env_kwargs):
    env = env_fn.env(**env_kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=config.illegal_reward)
    env = SB3ActionMaskWrapper(env)
    env.reset()  # Must call reset() in order to re-define the spaces
    if config.cuda:
        device = "cuda"
    else:
        device = "cpu"
    if config.masking:
        env = ActionMasker(env, mask_fn)
        model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=verbose, batch_size=config.minibatch_size, device=device, n_epochs=config.n_epochs , policy_kwargs={"activation_fn": nn.Tanh, "net_arch": config.layer_sizes_0})
        model.set_random_seed()
    else:
        model = PPO("MlpPolicy", env, verbose=verbose, device=device, batch_size=config.minibatch_size, n_epochs=config.n_epochs, policy_kwargs={"activation_fn": nn.Tanh, "net_arch": config.layer_sizes_0})
        model.set_random_seed()
    return model

def train_model(model, steps):
    model.env.reset()
    #print(steps)
    model.learn(total_timesteps=steps)
def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()




def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()



if __name__ == "__main__":
    env_fn = connect_four_v3

    env_logger.EnvLogger.suppress_output()

    args = parse_args()

    env_kwargs = {}

    model = create_model(connect_four_v3, args, True )
    for e in range(args.epochs):
        print(f"training epoch {e}")
        train_model(model, args.limit_steps_in_epoch)


