use crate::env::{SequentialGameState, GameTrajectory};
use crate::scheme::Scheme;


/// Environment that provide tracing game.
pub trait TracingEnvironment<DP: Scheme, ST: SequentialGameState<DP>>{

    
    fn trajectory(&self) -> &GameTrajectory<DP, ST>;

}