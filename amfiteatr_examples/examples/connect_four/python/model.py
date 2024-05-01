# based on https://gist.github.com/cyoon1729

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pettingzoo.utils import wrappers
from pettingzoo.utils import env_logger
from pettingzoo.classic import connect_four_v3
import logging

global dtype
dtype  = torch.FloatTensor

global gamma
gamma = 0.9

global logger

env_logger.EnvLogger.suppress_output()

def my_env(illegal_reward = -10, **kwargs):
    env = connect_four_v3.raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=illegal_reward)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
def parse_args():
    parser = argparse.ArgumentParser(description = "Baseline A2C model for ConnectFour Game")
    parser.add_argument('-e', "--epochs", type=int, default=100, help="Number of epochs of training")
    parser.add_argument('-g', "--games", type=int, default=128, help="Number of games in epochs of training")
    parser.add_argument('-t', "--test_games", type=int, default=100, help="Number of games in epochs of testing")
    parser.add_argument('-p', "--penalty", type=float, default=-10, help="NPenalty for illegal actions")
    parser.add_argument("--layer_sizes_1", metavar="LAYERS", type=int, nargs="*", default=[128, 256, 128], help = "Sizes of subsequent linear layers")
    parser.add_argument("--layer_sizes_2", metavar="LAYERS", type=int, nargs="*", default=[128, 256, 128],
                        help="Sizes of subsequent linear layers")


    return parser.parse_args()

class Agent:
    def __init__(self, id, policy, learning_rate = 1e-4):
        self.id = id
        self.policy = policy

        #self.observation = None

        self.current_trajectory = []
        self.batch_trajectories = []
        self.termianted = False
        self.optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)

    def select_action_based_on_observation(self, observation, score_pre_action):

        state_tensor =  Variable(torch.from_numpy(observation.flatten()).float())
        #distr, _critic = self.policy.forward(observation.flatten(), False)
        distr, _critic = self.policy.forward(state_tensor, False)

        selection = torch.multinomial(distr, 1).squeeze().item()

        self.current_trajectory.append((state_tensor, selection, score_pre_action))

        return selection

    def finish(self, score):
        if not self.termianted:
            self.current_trajectory.append((None, None, score))
            self.termianted = True


    def reset(self):
        self.current_trajectory = []
        self.termianted = False

    def clear_batch_trajectories(self):
        self.current_trajectory = []
        self.batch_trajectories = []

    def store_trajectory(self):

        self.batch_trajectories.append(self.current_trajectory)

    def train_network(self, trajectories):
        states = []
        discounted_rewards = []
        actions = []
        for t in trajectories:
            trajectory_states = [x[0] for x in t[:-1]]
            trajectory_discounted_rewards = [0]*(len(t) -1)
            actions_chosen = [torch.tensor([x[1]], dtype=torch.int64) for x in t[:-1]]

            current_discounted_reward = t[-1][2]
            trajectory_discounted_rewards[-1] = torch.Tensor([current_discounted_reward])

            # [0, 1, ..., -3, -2, -1]
            # -1 because we start from the last reward (index -1)
            # -1 because trajectory has last field (None, None, final reward)
            # -1 because final reward is done explicite above
            for r_index in range(len(t)-3, -1, -1):
                current_discounted_reward = t[r_index][2] + (gamma * current_discounted_reward)
                trajectory_discounted_rewards[r_index] = torch.Tensor([current_discounted_reward])


            states.extend(trajectory_states)
            discounted_rewards.extend(trajectory_discounted_rewards)
            actions.extend(actions_chosen)

        t_states = torch.stack(states, dim=0)
        t_discounted_rewards = torch.stack(discounted_rewards, dim=0)
        t_actions = torch.stack(actions, dim=0)

        t_actor, t_critic = self.policy.forward(t_states, True)



        # print(t_states.shape)
        # print(t_discounted_rewards.shape)
        # print(t_actions.shape)
        # print(t_actor.shape)
        # print(t_critic.shape)

        log_probs = t_actor.log_softmax(-1)
        probs = t_actor.softmax(-1)

        action_log_probs = log_probs.gather(1, t_actions)
        dist_entropy = (-log_probs * probs).sum()

        advantages = t_discounted_rewards - t_critic
        value_loss = (advantages * advantages).mean()
        action_loss = (-advantages * action_log_probs).mean()
        self.optimizer.zero_grad()
        loss = (value_loss * 0.5) + action_loss - dist_entropy * 0.01
        loss.backward()

        self.optimizer.step()

        #parameters = self.parameters()



class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_layers):
        super(ActorCritic, self).__init__()

        #self.optimizer = optim.Adam(lr=learning_rate)

        self.num_actions = num_actions

        self.layers = []

       # i = 1

        last_dim = num_inputs
        for layer_size in hidden_layers:
            #lin =
            self.layers.append(nn.Linear(last_dim, layer_size))
            self.layers.append(F.tanh)
            last_dim = layer_size

        if len(self.layers) > 0:
            self.hidden = True
        else:
            self.hidden = False

        #self.layers = nn.Sequential(*self.layers)


        self.ciritic_final = nn.Linear(last_dim, 1)
        self.actor_final = nn.Linear(last_dim, num_actions)


    def _net(self, value):


        #value = Variable(torch.from_numpy(state).float().unsqueeze(0))

        for layer in self.layers:
            value = layer(value)
        #    value = F.tanh(value)

        #self.layers(value)

        critic = self.ciritic_final(value)
        actor = F.softmax(self.actor_final(value), dim=0)
        return actor, critic



    def forward(self, state_tensor, use_grad=True):

        if use_grad:
            return self._net(state_tensor)
        else:
            with torch.no_grad():
                return self._net(state_tensor)








class Model:

    def other_agent(self, agent):
        if agent == self.agent0_id:
            return self.agent1_id
        else:
            return self.agent0_id


    def __init__(self, env, agent0, agent1):
        self.env = env
        self._prev_action_mask = None
        #self.agent0 = agent0
        #self.agent1 = agent1
        self.agent0_id = agent0.id
        self.agent1_id = agent1.id
        #self.agent1 = agent1
        #self.agent2 = agent2
        self.agents = {agent0.id: agent0, agent1.id: agent1}

    def play_single_game(self, store_episode, illegal_reward):
        #reinit environment because if illegal action is made then following episodes would fail on first or second action (when I simply reset), I don't know why
        #self.env = connect_four_v3.env(render_mode=None)
        self.env = my_env(render_mode=None, illegal_reward=illegal_reward)
        self.env.reset()
        self.env._prev_obs = None
        #print(self.env.board)
        self.agents[self.agent0_id].reset()
        self.agents[self.agent1_id].reset()
        winner = None
        cheater = None
        agent = None

        for agent in self.env.agent_iter():
            #print(agent)
            observation, reward, termination, truncation, info = self.env.last()

            #print(reward)

            if termination or truncation:
                action = None
                if reward > 0.0:
                    self.agents[agent].finish(1.0)
                    self.agents[self.other_agent(agent)].finish(-1.0)
                    winner = agent
                elif reward == -1.0:
                    self.agents[agent].finish(-1.0)
                    self.agents[self.other_agent(agent)].finish(1.0)
                    winner = self.other_agent(agent)
                elif reward < 0.0:
                    self.agents[agent].finish(illegal_reward)
                    self.agents[self.other_agent(agent)].finish(1.0)
                    cheater = agent
                else:
                    self.agents[agent].finish(0.0)
                    self.agents[self.other_agent(agent)].finish(0.0)
            else:
                state = observation["observation"]
                #t = torch.from_numpy(state).float()
                #print(t)
                mask = observation["action_mask"]
                #print("mask: ", mask)
                action = self.agents[agent].select_action_based_on_observation(state, 0.0)

            self.env.step(action)


        # print([(x[1],x[2]) for x in self.agents[self.agent0_id].current_trajectory])
        # print("||")
        # print([(x[1],x[2]) for x in self.agents[self.agent1_id].current_trajectory])

        if store_episode:
            self.agents[self.agent0_id].store_trajectory()
            self.agents[self.agent1_id].store_trajectory()

        return winner, cheater


    def play_epoch(self, number_of_games, summarize, is_training, illegal_reward):


        self.agents[self.agent0_id].clear_batch_trajectories()
        self.agents[self.agent1_id].clear_batch_trajectories()
        wins = {None: 0, self.agent0_id: 0, self.agent1_id:0 }
        cheats = {None: 0, self.agent0_id: 0, self.agent1_id: 0}

        for g in range(number_of_games):
            #print(f"game: {g}")
            winner, cheater = self.play_single_game(is_training, illegal_reward)
            wins[winner] = wins[winner] + 1
            cheats[cheater] = cheats[cheater] + 1



        return wins, cheats

    def apply_experience(self):
        t_agent0 = self.agents[self.agent0_id].batch_trajectories
        self.agents[self.agent0_id].train_network(t_agent0)
        t_agent1 = self.agents[self.agent1_id].batch_trajectories
        self.agents[self.agent1_id].train_network(t_agent1)







def main():

    args = parse_args()
    #print(args.layers1)
    #print(args.layers2)
    env = my_env(render_mode=None)
    env.reset()
    policy0 = ActorCritic(84, 7, args.layer_sizes_1 )
    agent0 = Agent(env.agents[0], policy0, learning_rate=1e-4)

    policy1 = ActorCritic(84, 7, args.layer_sizes_2)
    agent1 = Agent(env.agents[1], policy1, learning_rate=1e-4)

    agents = {agent0.id: agent0, agent1.id: agent1}
    model = Model(env, agent0, agent1)

    #model.play_single_game(True)


    #s = model.play_epoch(10, True, True, illegal_reward=-10.0)
    #print(s)
    #model.apply_experience()
    #t0 = model.agents[]
    #model.agents[agent0].policy.train_network()

    print("")
    w,c = model.play_epoch(args.test_games, True, False, args.penalty)
    print(f"Test before learning: wins: {w}, cheats: {c}")
    for e in range(args.epochs):
        model.play_epoch(args.test_games, False, True, args.penalty)
        model.apply_experience()
        w, c = model.play_epoch(args.test_games, True, False, args.penalty)
        print(f"Test after epoch {e}: wins: {w}, cheats: {c}")






    env.close()
    pass


if __name__ == "__main__":
    main()