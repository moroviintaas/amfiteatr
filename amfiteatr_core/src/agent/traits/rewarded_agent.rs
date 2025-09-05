use crate::scheme::{Scheme, Reward};

/// Trait for agents collecting rewards sent them by environment.
///
pub trait RewardedAgent<S: Scheme>{
    /// Returns currently stored universal reward
    /// This should be sum of partial rewards
    /// received by agent since his last action.
    /// With multiple agents environment can distribute rewards many times between
    /// actions of particular agent.
    /// > It might be easier to implement small rewards for every agent everytime any agent
    /// > performs action. Let's say agent _A_ performs action and based on change of state informs
    /// > agent _A_ that his reward is +3. Then play agents _B_ and _C_, and while their actions
    /// > change game state it generates partial rewards for _A_ respectively 0 and -1.
    /// > Score for this action (difference between state value before next and this action)
    /// > will be `+3 +0 -1 = +2`.
    /// > Agents will usually be interested in score change between __their__ subsequent actions.
    /// > Action that gives `+3` score but immediately is punished by other player with `-1` to `-2`
    /// > will be better than action giving `+10` and immediately punished by `-100`, right?.
    /// > So this should return score change since last action performed by this agent.
    fn current_universal_reward(&self) -> S::UniversalReward;
    //fn set_current_universal_reward(&mut self, reward: S::UniversalReward);
    /// Add partial reward to currently stacked reward since last action.
    fn current_universal_reward_add(&mut self, reward_fragment: &S::UniversalReward);
    /// Return score since the beginning to this moment. Usually it will be sum
    /// of _committed_ score and current partial rewards.
    fn current_universal_score(&self) -> S::UniversalReward;
    /// Commits current partial rewards to locked score and resets partial reward to
    /// neutral value.
    fn commit_partial_rewards(&mut self);

    /// Sets current payoff of agent, however does not change commited values (as they are allowed
    /// to commit only on certain points of steps).
    /// Instead, the uncommited part is changed accordingly.
    /// So lets say that score commited before taking action is 11.0, and since action was taken
    /// the score reduced to 8.0 (commited value is still 11.0, partial rewards are summed to -3.0).
    /// Now for some reason you don't want to add new partial reward 1.0, but want to arbitrarily
    /// set score to 9.0. Then call [`current_universal_score_set_without_commit(9.0)`].
    /// After that the commited score is still 11.0, however uncommited is `-3.0 + ((9.0 - 11.0) - (-3.0))`.
    /// Uncommited partial rewards are -2.0
    /// Implementation is provided.
    /// ```
    /// use amfiteatr_core::agent::{AgentGen, RandomPolicy, RewardedAgent};
    /// use amfiteatr_core::comm::StdAgentEndpoint;
    /// use amfiteatr_core::demo::{DemoDomain, DemoInfoSet};
    /// let mut agent = AgentGen::new(DemoInfoSet::new(0, 3), StdAgentEndpoint::new_pair().0, RandomPolicy::new());
    /// agent.current_universal_reward_add(&11.0);
    /// agent.commit_partial_rewards();
    /// agent.current_universal_reward_add(&-3.0);
    /// assert_eq!(agent.current_universal_reward(), -3.0);
    /// assert_eq!(agent.current_universal_score(), 8.0);
    /// agent.current_universal_score_set_without_commit(&9.0);
    /// assert_eq!(agent.current_universal_reward(), -2.0);
    /// assert_eq!(agent.current_universal_score(), 9.0);
    /// ```
    fn current_universal_score_set_without_commit(&mut self, payoff: &S::UniversalReward){
        let current_score_total = self.current_universal_score(); // 8.0 // (11.0 - 3.0)
        let difference = payoff // 9.0
            .ref_sub(&current_score_total); // 1.0 // 9.0 - (11.0 - 3.0)

            //.ref_sub(&self.current_universal_reward()); // 1.0
        self.current_universal_reward_add(&difference); //
    }
    fn current_universal_score_set_and_commit(&mut self, payoff: &S::UniversalReward){
        self.current_universal_score_set_without_commit(payoff);
        self.commit_partial_rewards();
    }

}