import torch


class Agent:
    def __init__(self, id, policy):
        self.id = id
        self.policy = policy


        #self.observation = None

        self.current_trajectory = []
        self.batch_trajectories = []
        self.terminated = False

        self.accumulated_reward_from_start = 0

        self.info_sets = []
        self.actions = []
        self.scores_before_step = []
        self.terminations = []
        self.truncations = []
        self.infos = []
        self.masks = []

        self.current_trajectory = []

    def policy_step(self, observation, score_before_step,  mask):

        #print(mask)
        observation_t = torch.from_numpy(observation.flatten()).float().to(self.policy.device)

        selection = self.policy.select_action(observation, None)[0]
        #print("selection:", selection, type(selection))

        self.info_sets.append(observation_t)
        self.scores_before_step.append(score_before_step)
        #self.terminations.append(termination)
        #self.truncations.append(truncation)
        self.masks.append(torch.from_numpy(mask).to(self.policy.device))
        self.actions.append(torch.tensor(selection).to(self.policy.device))



        return selection

    def finish(self, score):
        if not self.terminated:
            self.scores_before_step.append(score)
            self.terminated = True


    def reset(self):
        self.current_trajectory = []
        self.terminated = False
        self.accumulated_reward_from_start = 0
        self.info_sets = []
        self.actions = []
        self.scores_before_step = []
        self.truncations = []
        self.infos = []
        self.terminations = []
        self.masks = []

    def rewards(self):
        rewards = [self.scores_before_step[i+1] - self.scores_before_step[i] for i in range(len(self.scores_before_step)-1)]
        #print("rewards: ", rewards)
        return rewards

    def clear_batch_trajectories(self):
        #self.current_trajectory = []
        self.batch_trajectories = []
        self.info_sets = []
        self.actions = []
        self.scores_before_step = []
        self.truncations = []
        self.infos = []
        self.terminations = []
        self.masks = []

    def store_trajectory(self):

        #print(f"storing rewards {self.id}: ",  self.rewards())

        self.batch_trajectories.append((self.info_sets, self.actions, self.rewards(), self.masks))

        #print("bt", len(self.batch_trajectories), [x[2] for x in self.batch_trajectories])