import torch
from torch import nn

from c4.a2c import NetA2C
from torch.distributions.categorical import Categorical
import numpy as np
"""
This Policy is based on cleanrl implementation.
You can find implementation here: https://github.com/vwxyzjn/cleanrl/blob/v1.0.0/cleanrl/ppo.py
This modification splits action selection and train to different functions.
However we reduce number of environments to 1, because we collect trajectories for every agent separately,
and therefore they might desynchronize when some environments end earlier than other.
"""
class PolicyPPO:
    def __init__(self, num_inputs, num_actions, hidden_layers, config, masking=False, device=None, tb_writer=None):
        ##__init__(self, num_inputs, num_actions, hidden_layers, config, action_space, device=None, ):
        #super(ActorCritic, self).__init__()
        self.device = device

        self.config = config
        self.network = NetA2C(num_inputs, num_actions, hidden_layers, device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = config.learning_rate)
        self.tb_writer = tb_writer
        self.global_step = 0
        self.masking = masking

        #self.action_space = action_space

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

    def get_action_and_value(self, x, action=None, masks=None):
        actor, critic = self.network.forward(x)
        #logits = self.actor(x)
        # We have defined singe actor,critic function (run network once)
        if masks is None or self.masking == False:
            logits = actor
        else:

            mask_value = torch.tensor(torch.finfo(actor.dtype).min, dtype=actor.dtype)
            logits = torch.where(masks, actor, mask_value)
        #print(f"logits_shape: {logits.shape}")

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        #print(f"Action_t: ({action.shape}) {action}, ")
        #print(f"ALP: ({probs.log_prob(action.squeeze(-1)).shape}, {probs.log_prob(action.squeeze(-1))}")
        return action, probs.log_prob(action.squeeze(-1)), probs.entropy(), critic
        #return action, probs.log_prob(action), probs.entropy(), critic




    def train_network(self, trajectories):

        example_step = trajectories[0][0]
        #print(len(trajectories))
        #print(len(trajectories[0]))
        #print(len(example_step))
        #e_obs, e_action, e_reward, e_masks = example_step
        e_obs = trajectories[0][0][0]
        e_action = trajectories[0][1][0]
        num_steps = sum(len(t) for t in trajectories)

        device = self.device
        num_envs = 1
        single_observation_space = e_obs
        single_action_space = e_action


        batch_obs = []
        batch_logprobs = []
        batch_actions = []
        batch_advantages = []
        batch_returns = []
        batch_values = []
        batch_masks = []

        value_loses = []
        entropies = []
        policy_gradient_losses = []

        #obs = torch.zeros((num_steps, num_envs) + single_observation_space.shape).to(device)
        #actions = torch.zeros((num_steps, num_envs) + single_action_space.shape).to(device)
        #logprobs = torch.zeros((num_steps, num_envs)).to(device)
        #rewards = torch.zeros((num_steps, num_envs)).to(device)
        #dones = torch.zeros((num_steps, num_envs)).to(device)
        #values = torch.zeros((num_steps, num_envs)).to(device)

        for (states, actions, rewards, masks) in trajectories:
            if len(states) == 0:
                break
            states_tensor = torch.stack(states, dim=0)
            actions_tensor = torch.stack(actions, dim=0)


            rewards_tensor = torch.stack([torch.tensor(r) for r in rewards], dim =0)
            with torch.no_grad():
                #_actor, critic = self.network.forward(states_tensor,  masks_tensor)
                if self.masking:
                    masks_tensor = torch.stack(masks, dim=0)
                    batch_masks.append(masks_tensor)
                else:
                    masks_tensor = None

                _action, logprob, entropy, critic = self.get_action_and_value(states_tensor, actions_tensor, masks_tensor)
                advantages = torch.zeros((len(rewards),1)).to(self.device)
                lastgaelam = 0
                for t in reversed(range(len(states))):
                    if t == len(states) -1:
                        nextvalue = 0.0
                        nextnoterminal = 0.0
                    else:
                        nextnoterminal = 1.0
                        nextvalue = critic[t+1]
                    #print(type(self.config.gamma))
                    #print(type(self.config.gae_lambda))
                    #print(type(nextvalue))
                    #print(type(rewards[t]))
                    delta = rewards[t] + (self.config.gamma * nextvalue * nextnoterminal) - critic[t]
                    advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnoterminal * lastgaelam


                returns = advantages + critic
                # print("r:", rewards)
                # print("A:", advantages)
                # print("C:", critic)
                # print("R:", returns)


            batch_obs.append(states_tensor)
            batch_logprobs.append(logprob.unsqueeze(1))
            batch_actions.append(actions_tensor.unsqueeze(1))
            batch_advantages.append(advantages.unsqueeze(1))
            batch_returns.append(returns)
            batch_values.append(critic)



        b_obs = torch.vstack(batch_obs)
        #for t in batch_logprobs:
        #    print(t.size())
        b_logprobs = torch.vstack(batch_logprobs).squeeze(-1)
        b_actions = torch.vstack(batch_actions)
        b_advantages = torch.vstack(batch_advantages)
        b_returns = torch.vstack(batch_returns)
        b_values = torch.vstack(batch_values)

        batch_size = b_obs.size()[0]
        b_inds = np.arange(batch_size)
        if self.masking:
            b_masks = torch.vstack(batch_masks)
        else:
            b_masks = None
        clipfracs = []


        #minibatch_size = int(batch_size // self.config.num_minibatches)

        #print(minibatch_size)


        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self.config.minibatch_size):
                end = start +  self.config.minibatch_size
                mb_inds = b_inds[start:end]

                if self.masking:
                    _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], b_masks[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                #print(f"Newlogprob: {newlogprob.shape}, b_logprobs: {b_logprobs[mb_inds].shape}")
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                #print(f"A: {advantages.shape}, ratio: {ratio.shape}")
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                        )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.entropy_coef * entropy_loss + v_loss * self.config.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()



            #if self.config.target_kl is not None and approx_kl > self.config.target_kl:
            #    break
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.tb_writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.tb_writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.tb_writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            #self.tb_writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
            self.tb_writer.add_scalar("losses/kl_approximation", approx_kl.item(), self.global_step)
            #self.tb_writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.global_step += 1
            #self.tb_writer.add_scalar("losses/explained_variance", explained_var, global_step)





                

