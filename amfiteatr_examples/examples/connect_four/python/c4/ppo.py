import torch
from torch import nn

from c4.a2c import NetA2C



"""
This Policy is based on cleanrl implementation.
You can find implementation here: https://github.com/vwxyzjn/cleanrl/blob/v1.0.0/cleanrl/ppo.py
This modification splits action selection and train to different functions.
However we reduce number of environments to 1, because we collect trajectories for every agent separately,
and therefore they might desynchronize when some environments end earlier than other.
"""
class PolicyPPO:
    def __init__(self, num_inputs, num_actions, hidden_layers, config, action_space, device=None, ):
        #super(ActorCritic, self).__init__()
        self.device = device

        self.config = config
        self.network = NetA2C(num_inputs, num_actions, hidden_layers, device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = config.learning_rate)
        self.action_space = action_space

    def select_action(self, info_set, masks):
        info_set_tensor = torch.from_numpy(info_set.flatten()).float().to(self.device)

        actor, _critic = self.network.forward(info_set_tensor)

        if masks is None:
            dist = actor.softmax(-1)
        else:
            print("Masks not supported yet")
        #print("actor = ", actor)
        #print("dist = ", dist)


        selection = torch.multinomial(dist, 1).item()

        return selection,


    def get_masked_actor_and_critic(self, info_set_tensor, mask_tensor):
        actor, critic = self.network.forward(info_set_tensor)
        dist = actor.softmax(-1)

        if mask_tensor is not None:
            #dist = actor.
            pass


        return dist, critic


    def train_network(self, trajectories):

        example_step = trajectories[0][0]
        e_obs, e_action, e_reward, e_masks = example_step

        num_steps = sum(len(t) for t in trajectories)

        device = self.device
        num_envs = 1
        single_observation_space = e_obs
        single_action_space = e_action




        #obs = torch.zeros((num_steps, num_envs) + single_observation_space.shape).to(device)
        #actions = torch.zeros((num_steps, num_envs) + single_action_space.shape).to(device)
        #logprobs = torch.zeros((num_steps, num_envs)).to(device)
        #rewards = torch.zeros((num_steps, num_envs)).to(device)
        #dones = torch.zeros((num_steps, num_envs)).to(device)
        #values = torch.zeros((num_steps, num_envs)).to(device)

        for (states, actions, rewards, masks) in trajectories:
            states_tensor = torch.stack(states, dim=0)
            masks_tensor = torch.stack(masks, dim=0)
            rewards_tensor = torch.stack(rewards, dim =0)
            with torch.no_grad():
                _actor, critic = self.network.forward(states_tensor,  masks_tensor)
                advantages = torch.zeros(len(rewards))
                lastgaelam = 0
                for t in reversed(range(len(states))):
                    if t == len(states) -1:
                        nextvalue = 0.0
                        nextnoterminal = 0.0
                    else:
                        nextnoterminal = 1.0
                        nextvalue = critic[t+1]
                    delta = rewards[t] + self.config.gamma * nextvalue * nextnoterminal * lastgaelam
                    advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnoterminal * lastgaelam
                returns = advantages + critic
                

