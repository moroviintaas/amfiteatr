use crate::env::{SequentialGameState, GameTrajectory};
use crate::domain::DomainParameters;


/// Environment that provide tracing game.
pub trait TracingEnvironment<DP: DomainParameters, S: SequentialGameState<DP>>{

    
    fn trajectory(&self) -> &GameTrajectory<DP, S>;

}