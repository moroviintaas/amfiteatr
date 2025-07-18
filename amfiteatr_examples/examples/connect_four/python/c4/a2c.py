import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np


class NetA2C(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_layers, device=None):
        super(NetA2C, self).__init__()
        last_dim = num_inputs
        if hidden_layers is not None:
            layers = []
            for layer_size in hidden_layers:
                layers.append(nn.Linear(last_dim, layer_size, device=device))
                layers += [nn.Tanh()]
                last_dim = layer_size
            self.seq = nn.Sequential(*layers)

        else:
            self.seq = None
        self.actor = nn.Linear(last_dim, num_actions, device=device)
        self.critic = nn.Linear(last_dim, 1, device=device)

    def forward(self, state_tensor):
        if self.seq is not None:
            s = self.seq(state_tensor)
            return self.actor(s), self.critic(s)
        else:
            return self.actor(state_tensor), self.critic(state_tensor)





class PolicyA2C:
    def __init__(self, num_inputs, num_actions, hidden_layers, config, masking=False, device=None, tb_writer=None):
        #super(ActorCritic, self).__init__()
        self.device = device

        self.config = config
        self.network = NetA2C(num_inputs, num_actions, hidden_layers, device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = config.learning_rate)
        self.tb_writer = tb_writer
        self.masking = masking

    def select_action(self, info_set, masks):
        info_set_tensor = torch.from_numpy(info_set.flatten()).float().to(self.device)

        with torch.no_grad():
            actor, _critic = self.network.forward(info_set_tensor)

        if masks is None:
            probs = Categorical(logits=actor)
        else:
            mask_value = torch.tensor(torch.finfo(actor.dtype).min, dtype=actor.dtype)
            logits = torch.where(masks, actor, mask_value)
            probs = Categorical(logits=logits)

        selection  =probs.sample()

        return selection


    def get_masked_actor_and_critic(self, info_set_tensor, mask_tensor):
        actor, critic = self.network.forward(info_set_tensor)
        dist = actor.softmax(-1)

        if mask_tensor is not None:
            #dist = actor.
            pass


        return dist, critic

    def train_network(self, trajectories):

        batch_states = []
        batch_actions = []
        batch_discounted_rewards = []
        batch_masks = []

        #print("t: ", trajectories[0][2])
        #print("t: ", trajectories[0][1])

        for (states, actions, rewards, masks) in trajectories:
            if len(rewards) == 0:
                continue
            #print("rew in t", rewards)
            discounted_rewards = [0] * (len(rewards))
            discounted_rewards[-1] = rewards[-1]
            for i in range(len(rewards)-2, -1, -1):
                discounted_rewards[i] = rewards[i] + (self.config.gamma * discounted_rewards[i+1])

            #print("Discounted rewards: ", discounted_rewards)
            discounted_rewards_t = torch.Tensor(discounted_rewards)
            #masks_t = torch.stack(masks, dim=0)

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_discounted_rewards.extend(discounted_rewards_t)
            batch_masks.extend(masks)

        t_states = torch.stack(batch_states, dim=0).to(self.device)
        t_discounted_rewards = torch.stack(batch_discounted_rewards, dim=0).to(self.device)
        t_actions = torch.stack(batch_actions, dim=0).to(self.device).unsqueeze(-1)
        #t_masks = torch.stack(batch_masks, dim=0).to(self.device)
        if self.masking:

            t_masks = torch.stack(batch_masks, dim=0).to(self.device)
        else:
            t_masks = None




        # t_actor, t_critic = self.network.forward(t_states)
        # if self.masking and t_masks is not None:
        #     mask_value = torch.tensor(torch.finfo(t_actor.dtype).min, dtype=t_actor.dtype)
        #     log_probs = torch.where(t_masks, t_actor, mask_value).log_softmax(-1)
        # else:
        #     log_probs = t_actor.log_softmax(-1)
        # probs = log_probs.exp()


        batch_size = t_states.shape[0]
        b_inds = np.arange(batch_size)
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, self.config.minibatch_size):

            end = start +  self.config.minibatch_size
            mb_inds = b_inds[start:end]
            #print(mb_inds)

            mb_actor, mb_critic = self.network.forward(t_states[mb_inds])
            if self.masking and t_masks is not None:
                mask_value = torch.tensor(torch.finfo(mb_actor.dtype).min, dtype=mb_actor.dtype)
                log_probs = torch.where(t_masks[mb_inds], mb_actor, mask_value).log_softmax(-1)
            else:
                log_probs = mb_actor.log_softmax(-1)
            probs = log_probs.exp()

            action_log_probs = log_probs.gather(1, t_actions[mb_inds])

            entropy = (-log_probs * probs).sum(1).mean()

            advantages = t_discounted_rewards[mb_inds] - mb_critic
            value_loss = (advantages * advantages).mean()
            action_loss  = -(advantages * action_log_probs).mean()

            entropy_loss_scaled = entropy * self.config.entropy_coef
            loss = (value_loss * self.config.vf_coef) + action_loss - entropy_loss_scaled
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        # action_log_probs = log_probs.gather(1, t_actions)
        # entropy = (-log_probs * probs).sum(1).mean()
        # advantages = t_discounted_rewards - t_critic
        # value_loss = (advantages * advantages).mean()
        # action_loss  = -(advantages * action_log_probs).mean()


        # entropy_loss_scaled = entropy * self.config.entropy_coef
        # loss = (value_loss * self.config.vf_coef) + action_loss - entropy_loss_scaled
        # self.optimizer.zero_grad()
        # loss.backward()

        #self.optimizer.step()

        return {"value_loss": value_loss.item(), "action_loss": action_loss.item(), "entropy": entropy.item()}





