use crate::env::{EnvironmentStateSequential, EnvironmentTrajectory};
use crate::domain::DomainParameters;


/// Environment that provide tracing game.
pub trait TracingEnvironment<DP: DomainParameters, S: EnvironmentStateSequential<DP>>{

    
    fn trajectory(&self) -> &EnvironmentTrajectory<DP, S>;

}