use crate::env::{SequentialGameState, GameTrajectory};
use crate::scheme::Scheme;


/// Environment that provide tracing game.
pub trait TracingEnvironment<S: Scheme, ST: SequentialGameState<S>>{

    
    fn trajectory(&self) -> &GameTrajectory<S, ST>;

}