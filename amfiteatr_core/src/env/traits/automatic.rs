

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
pub trait AutoEnvironment<S: Scheme>{
    /// This method is meant to automatically run game and communicate with agents
    /// until is the game is finished.
    /// This method is not required to send agents messages with their scores.
    /// Argument `truncate_steps` determines if game should be truncated after certain number of steps.
    /// With `None` no truncation is made.
    ///
    /// Returns the number of all game steps made.
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>>;
    /// Just like [`run_truncating`]`(None`), left for compatibility reasons.
    fn run(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.run_truncating(None)
    }
}

/// Trait for environment automatically running a game with informing agents about their
/// rewards during game.
pub trait AutoEnvironmentWithScores<S: Scheme>{
    /// Method analogous to [`AutoEnvironment::run`](AutoEnvironment::run),
    /// but it should implement sending rewards to agents.
    /// Argument `truncate_steps` determines if game should be truncated after certain number of steps.
    /// With `None` no truncation is made.
    ///
    /// Returns the number of all game steps made.
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>>;

    /// Just like [`run_with_scores_truncating`]`(None)`, left for compatibility reasons.
    fn run_with_scores(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.run_with_scores_truncating(None)
    }
    //fn run_with_scores_and_penalties<P: Fn(&S::AgentId) -> S::UniversalReward>(&mut self, penalty: P) -> Result<(), AmfiError<S>>;
}
/// Trait for environment automatically running a game with informing agents about their
/// rewards during game and applying penalties to agents who
/// perform illegal (wrong) actions.
pub trait AutoEnvironmentWithScoresAndPenalties<S: Scheme>: StatefulEnvironment<S>{
    fn run_with_scores_and_penalties_truncating<P: Fn(&<Self as StatefulEnvironment<S>>::State, &S::AgentId)
        -> S::UniversalReward>(&mut self, penalty: P, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>>;

    /// Returns the number of all game steps made.
    ///
    /// Just like [`run_with_scores_and_penalties_truncating`]`(None)`, left for compatibility reasons.
    fn run_with_scores_and_penalties<P: Fn(&<Self as StatefulEnvironment<S>>::State, &S::AgentId)
        -> S::UniversalReward>(&mut self, penalty: P) -> Result<(), AmfiteatrError<S>>{
        self.run_with_scores_and_penalties_truncating(penalty, None)
    }
}


pub(crate) trait AutoEnvInternals<S: Scheme>{
    fn notify_error(&mut self, error: AmfiteatrError<S>) -> Result<(), CommunicationError<S>>;
    fn send_message(&mut self, agent: &S::AgentId, message: EnvironmentMessage<S>) -> Result<(), CommunicationError<S>>;
    //fn process_action_and_inform(&mut self, player: S::AgentId, action: &S::ActionType) -> Result<(), AmfiteatrError<S>>;
}

impl <
    S: Scheme,
    E: StatefulEnvironment<S>
        + CommunicatingEnvironmentSingleQueue<S>
        + BroadcastingEnvironmentSingleQueue<S>
> AutoEnvInternals<S> for E{
    fn notify_error(&mut self, error: AmfiteatrError<S>) -> Result<(), CommunicationError<S>> {
        self.send_all(EnvironmentMessage::ErrorNotify(error))
    }

    fn send_message(&mut self, agent: &<S as Scheme>::AgentId, message: EnvironmentMessage<S>) -> Result<(), CommunicationError<S>> {
        self.send(agent, message)
            .inspect_err(|e|{
                self.notify_error(e.clone().into())
                    .unwrap_or_else(|_|panic!("Failed broadcasting error message {}", e));
            })
    }

}

