

use crate::{
    error::{
        AmfiteatrError,
        CommunicationError
    },
    scheme::{
        Scheme,
        EnvironmentMessage,
    },
};

use super::{StatefulEnvironment, CommunicatingEnvironmentSingleQueue, BroadcastingEnvironmentSingleQueue};

/// Trait for environment automatically running a game.
pub trait AutoEnvironment<DP: Scheme>{
    /// This method is meant to automatically run game and communicate with agents
    /// until is the game is finished.
    /// This method is not required to send agents messages with their scores.
    /// Argument `truncate_steps` determines if game should be truncated after certain number of steps.
    /// With `None` no truncation is made.
    ///
    /// Returns the number of all game steps made.
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>>;
    /// Just like [`run_truncating`]`(None`), left for compatibility reasons.
    fn run(&mut self) -> Result<(), AmfiteatrError<DP>>{
        self.run_truncating(None)
    }
}

/// Trait for environment automatically running a game with informing agents about their
/// rewards during game.
pub trait AutoEnvironmentWithScores<DP: Scheme>{
    /// Method analogous to [`AutoEnvironment::run`](AutoEnvironment::run),
    /// but it should implement sending rewards to agents.
    /// Argument `truncate_steps` determines if game should be truncated after certain number of steps.
    /// With `None` no truncation is made.
    ///
    /// Returns the number of all game steps made.
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>>;

    /// Just like [`run_with_scores_truncating`]`(None)`, left for compatibility reasons.
    fn run_with_scores(&mut self) -> Result<(), AmfiteatrError<DP>>{
        self.run_with_scores_truncating(None)
    }
    //fn run_with_scores_and_penalties<P: Fn(&DP::AgentId) -> DP::UniversalReward>(&mut self, penalty: P) -> Result<(), AmfiError<DP>>;
}
/// Trait for environment automatically running a game with informing agents about their
/// rewards during game and applying penalties to agents who
/// perform illegal (wrong) actions.
pub trait AutoEnvironmentWithScoresAndPenalties<DP: Scheme>: StatefulEnvironment<DP>{
    fn run_with_scores_and_penalties_truncating<P: Fn(&<Self as StatefulEnvironment<DP>>::State, &DP::AgentId)
        -> DP::UniversalReward>(&mut self, penalty: P, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>>;

    /// Returns the number of all game steps made.
    ///
    /// Just like [`run_with_scores_and_penalties_truncating`]`(None)`, left for compatibility reasons.
    fn run_with_scores_and_penalties<P: Fn(&<Self as StatefulEnvironment<DP>>::State, &DP::AgentId)
        -> DP::UniversalReward>(&mut self, penalty: P) -> Result<(), AmfiteatrError<DP>>{
        self.run_with_scores_and_penalties_truncating(penalty, None)
    }
}


pub(crate) trait AutoEnvInternals<DP: Scheme>{
    fn notify_error(&mut self, error: AmfiteatrError<DP>) -> Result<(), CommunicationError<DP>>;
    fn send_message(&mut self, agent: &DP::AgentId, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>>;
    //fn process_action_and_inform(&mut self, player: DP::AgentId, action: &DP::ActionType) -> Result<(), AmfiteatrError<DP>>;
}

impl <
    DP: Scheme,
    E: StatefulEnvironment<DP> 
        + CommunicatingEnvironmentSingleQueue<DP>
        + BroadcastingEnvironmentSingleQueue<DP>
> AutoEnvInternals<DP> for E{
    fn notify_error(&mut self, error: AmfiteatrError<DP>) -> Result<(), CommunicationError<DP>> {
        self.send_all(EnvironmentMessage::ErrorNotify(error))
    }

    fn send_message(&mut self, agent: &<DP as Scheme>::AgentId, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>> {
        self.send(agent, message)
            .inspect_err(|e|{
                self.notify_error(e.clone().into())
                    .unwrap_or_else(|_|panic!("Failed broadcasting error message {}", e));
            })
    }

}

