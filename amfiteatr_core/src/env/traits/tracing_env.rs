use crate::env::{EnvironmentStateSequential,  GameTrajectory};
use crate::domain::DomainParameters;


/// Environment that provide tracing game.
pub trait TracingEnvironment<DP: DomainParameters, S: EnvironmentStateSequential<DP>>{

    
    fn trajectory(&self) -> &GameTrajectory<DP, S>;

}