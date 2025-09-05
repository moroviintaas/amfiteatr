use crate::env::{SequentialGameState, StatefulEnvironment};
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;

/// Environment that has state and can evaluate payoff for every player.
pub trait ScoreEnvironment<S: Scheme>: StatefulEnvironment<S>{

    /// This is substitute method for
    /// [`process_action`](crate::env::StatefulEnvironment::process_action).
    /// If action performed was illegal penalty reward is sent to that player and
    /// state should not be changed. This behaviour is experimental concept,
    /// use it when agent may make illegal action, you want to note it for future analysis
    /// (i.e. for reinforcement learning). Game can be continued from previous state.
    fn process_action_penalise_illegal(
        &mut self,
        agent: &S::AgentId,
        action: &S::ActionType,
        penalty_reward: S::UniversalReward)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>>;

    /// Return actual payoff of player based on game state
    fn actual_state_score_of_player(&self, agent: &S::AgentId) -> S::UniversalReward;
    /// Return actual penalty accumulated of player
    fn actual_penalty_score_of_player(&self, agent: &S::AgentId) -> S::UniversalReward;
    /// Return actual payoff of player, sum of state related payoff and penalty
    fn actual_score_of_player(&self, agent: &S::AgentId) -> S::UniversalReward{
        self.actual_state_score_of_player(agent) + self.actual_penalty_score_of_player(agent)
    }



}