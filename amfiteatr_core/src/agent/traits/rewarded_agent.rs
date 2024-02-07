use crate::domain::DomainParameters;

/// Trait for agents collecting rewards sent them by environment.
///
pub trait RewardedAgent<DP: DomainParameters>{
    /// Returns currently stored universal reward
    /// This should be sum of partial rewards
    /// received by agent since his last action.
    /// With multiple agents environment can distribute rewards many times between
    /// actions of particular agent.
    /// > It might be easier to implement small rewards for every agent everytime any agent
    /// performs action. Let's say agent _A_ performs action and based on change of state informs
    /// agent _A_ that his reward is +3. Then play agents _B_ and _C_, and while their actions
    /// change game state it generates partial rewards for _A_ respectively 0 and -1.
    /// Score for this action (difference between state value before next and this action)
    /// will be `+3 +0 -1 = +2`.
    /// Agents usually will be interested in score change between __their__ subsequent actions.
    /// Action that gives `+3` score but immediately is punished by other player with `-1` to `-2`
    /// will be better than action giving `+10` and immediately punished by `-100`, right?.
    /// So this should return score change since last action performed by this agent.
    fn current_universal_reward(&self) -> DP::UniversalReward;
    //fn set_current_universal_reward(&mut self, reward: DP::UniversalReward);
    /// Add partial reward to currently stacked reward since last action.
    fn current_universal_reward_add(&mut self, reward_fragment: &DP::UniversalReward);
    /// Return score since the beginning to this moment. Usually it will be sum
    /// of _committed_ score and current partial rewards.
    fn current_universal_score(&self) -> DP::UniversalReward;
    /// Commits current partial rewards to locked score and resets partial reward to
    /// neutral value.
    fn commit_partial_rewards(&mut self);
}