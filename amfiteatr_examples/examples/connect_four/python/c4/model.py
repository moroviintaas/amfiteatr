from c4.common import *
from c4.agent import *

from c4.a2c import *

from pettingzoo.utils import wrappers
from pettingzoo.utils import env_logger
from pettingzoo.classic import connect_four_v3
from sb3_contrib.common.wrappers import ActionMasker


def my_env(illegal_reward = -10, **kwargs):
    env = connect_four_v3.raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=illegal_reward)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    #env = ActionMasker(env)
    return env


class TwoPlayerModel:

    def __init__(self, env, agent0, agent1, model_config):
        self.env = env
        self.agents_ids = [agent0.id, agent1.id]
        self.agents = {agent0.id: agent0, agent1.id: agent1}

        self.env_step_counter = 0
        #self.max_env_steps_in_epoch = model_config.max_env_steps_in_epoch
        self.masking = model_config.masking



    def other_agent(self, agent_id):
        if agent_id == self.agents_ids[0]:
            return  self.agents_ids[1]
        elif agent_id == self.agents_ids[1]:
            return self.agents_ids[0]


    def play_single_game(self, store_episode, illegal_reward, truncate_at_steps=None):

        self.env = my_env(render_mode=None, illegal_reward=illegal_reward)
        self.env.reset()
        self.env._prev_obs = None
        #print(self.env.board)
        self.agents[self.agents_ids[0]].reset()
        self.agents[self.agents_ids[1]].reset()
        winner = None
        cheater = None
        agent = None
        payoffs = {}
        step = 0
        current_cumulative_reward = {}

        global_truncation = False

        for agent in self.env.agent_iter():
            #print(agent, type(agent))
            observation, payoff, termination, truncation, info = self.env.last(agent)
            #current_cumulative_reward[agent] = payoff
            mask = self.env.observe(agent)["action_mask"]
            #print(self.env.observe(agent)["action_mask"])

            #truncation = truncation or global_truncation
            payoffs[agent] = payoff

            if termination or truncation :

                #print(f"Termination on player {agent}, payoff: {payoff}")
                ##if payoff not in (0.0, -1.0, 1.0):
                #print(payoff)


                self.agents[agent].finish(payoff)
                action = None
                #if payoff == 1.0:
                #    self.agents[agent].finish(1.0)
                #    self.agents[self.other_agent(agent)].finish(-1.0)
                #    winner = agent
                #elif payoff == -1.0:
                #    self.agents[agent].finish(-1.0)
                #    self.agents[self.other_agent(agent)].finish(1.0)
                #    winner = self.other_agent(agent)
                #else:
                #    print(agent)
                #    print(observation)
                #    print(payoff)
                #    #print(f"Illegal move of player {agent}, score = {illegal_reward}")
                #    self.agents[agent].finish(illegal_reward)
                #    self.agents[self.other_agent(agent)].finish(0.0)
                #    cheater = agent

            else:
                state = observation["observation"]
                if self.masking:
                    masks = observation["action_mask"]
                else:
                    masks = None
                action = self.agents[agent].policy_step(state, payoff, masks)
                #print(f"Agent {agent} places {action}")
                #print(self.env.action_space(agent))
                #print(action)
            #self.
            #if not global_truncation:
            self.env.step(action)
            step += 1
            if truncate_at_steps is not None:
                if step >= truncate_at_steps:
                    global_truncation = True

                    try:
                        _, payoff, _, _, _ = self.env.last(self.other_agent(agent))
                    except KeyError:
                        payoff = 0.0
                    payoffs[self.other_agent(agent)] = payoff
                    self.agents[self.other_agent(agent)].finish(payoff)
                    #self.agents[self.other_agent(agent)].finish(payoffs[self.other_agent(agent)])
                    try:
                        _, payoff, _, _, _ = self.env.last(agent)
                    except KeyError:
                        payoff = 0.0
                    payoffs[agent] = payoff
                    self.agents[agent].finish(payoff)
                    #self.agents[agent].finish(payoffs[agent])
                    break

        #_, payoff0, _, _, _ = self.env.last(self.agents_ids[0])
        #_, payoff1, _, _, _ = self.env.last(self.agents_ids[1])

        if payoffs[self.agents_ids[0]] > payoffs[self.agents_ids[1]]:
            winner = self.agents_ids[0]
        elif payoffs[self.agents_ids[1]] > payoffs[self.agents_ids[0]]:
            winner = self.agents_ids[1]
        else:
            winner = None

        if payoffs[self.agents_ids[0]] == illegal_reward:
            cheater = self.agents_ids[0]
        elif payoffs[self.agents_ids[1]] == illegal_reward:
            cheater = self.agents_ids[1]
        else:
            cheater = None


        if store_episode:
            #print("store is on")
            self.agents[self.agents_ids[0]].store_trajectory()
            self.agents[self.agents_ids[1]].store_trajectory()

        return winner, cheater, step

    def play_epoch(self, number_of_games, is_training, illegal_reward, limit_steps_in_epoch):

        games_played = 0
        total_steps = 0
        steps_left = limit_steps_in_epoch
        self.agents[self.agents_ids[0]].clear_batch_trajectories()
        self.agents[self.agents_ids[1]].clear_batch_trajectories()

        wins = {None: 0, self.agents_ids[0]: 0, self.agents_ids[1]: 0 }
        cheats = {None: 0, self.agents_ids[0]: 0, self.agents_ids[1]: 0}
        scores = {None: 0, self.agents_ids[0]: 0, self.agents_ids[1]: 0}

        for g in range(number_of_games):
            #print(steps_left)
            winner, cheater, game_steps = self.play_single_game(is_training, illegal_reward, steps_left)
            total_steps += game_steps
            games_played += 1
            wins[winner] = wins[winner] + 1
            #scores[winner] =
            cheats[cheater] = cheats[cheater] + 1

            if steps_left is not None:
                steps_left -= game_steps
                if steps_left <= 0:
                    break

        for (p, score) in wins.items():
            wins[p] = score/games_played
        for (p, illegal) in cheats.items():
            cheats[p] = illegal/games_played



        return wins, cheats, games_played, total_steps


    def apply_experience(self):
        report = {}
        for (id, agent) in self.agents.items():
            report[id] = agent.policy.train_network(agent.batch_trajectories)
        return report