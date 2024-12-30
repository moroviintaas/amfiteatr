

use crate::{
    error::{
        AmfiteatrError,
        CommunicationError
    },
    domain::{
        DomainParameters,
        EnvironmentMessage,
    },
};

use super::{StatefulEnvironment, CommunicatingAdapterEnvironment, BroadConnectedEnvironment};

/// Trait for environment automatically running a game.
pub trait AutoEnvironment<DP: DomainParameters>{
    /// This method is meant to automatically run game and communicate with agents
    /// until is the game is finished.
    /// This method is not required to send agents messages with their scores.
    fn run(&mut self) -> Result<(), AmfiteatrError<DP>>;
}

/// Trait for environment automatically running a game with informing agents about their
/// rewards during game.
pub trait AutoEnvironmentWithScores<DP: DomainParameters>{
    /// Method analogous to [`AutoEnvironment::run`](AutoEnvironment::run),
    /// but it should implement sending rewards to agents.
    fn run_with_scores(&mut self) -> Result<(), AmfiteatrError<DP>>;
    //fn run_with_scores_and_penalties<P: Fn(&DP::AgentId) -> DP::UniversalReward>(&mut self, penalty: P) -> Result<(), AmfiError<DP>>;
}
/// Trait for environment automatically running a game with informing agents about their
/// rewards during game and applying penalties to agents who
/// perform illegal (wrong) actions.
pub trait AutoEnvironmentWithScoresAndPenalties<DP: DomainParameters>: StatefulEnvironment<DP>{
    fn run_with_scores_and_penalties<P: Fn(&<Self as StatefulEnvironment<DP>>::State, &DP::AgentId)
        -> DP::UniversalReward>(&mut self, penalty: P) -> Result<(), AmfiteatrError<DP>>;
}


pub(crate) trait AutoEnvInternals<DP: DomainParameters>{
    fn notify_error(&mut self, error: AmfiteatrError<DP>) -> Result<(), CommunicationError<DP>>;
    fn send_message(&mut self, agent: &DP::AgentId, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>>;
    //fn process_action_and_inform(&mut self, player: DP::AgentId, action: &DP::ActionType) -> Result<(), AmfiteatrError<DP>>;
}

impl <
    DP: DomainParameters,
    E: StatefulEnvironment<DP> 
        + CommunicatingAdapterEnvironment<DP>
        + BroadConnectedEnvironment<DP>
> AutoEnvInternals<DP> for E{
    fn notify_error(&mut self, error: AmfiteatrError<DP>) -> Result<(), CommunicationError<DP>> {
        self.send_all(EnvironmentMessage::ErrorNotify(error))
    }

    fn send_message(&mut self, agent: &<DP as DomainParameters>::AgentId, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>> {
        self.send(agent, message)
            .map_err(|e|{
                self.notify_error(e.clone().into())
                    .unwrap_or_else(|_|panic!("Failed broadcasting error message {}", &e));
                e
            })
    }

}

