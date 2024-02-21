pub mod env;
pub mod common;
pub mod agent;
use amfiteatr_core::env::EnvironmentStateSequential;
use crate::common::SINGLE_PLAYER_ID;
use crate::env::PythonGymnasiumCartPoleState;


fn main() {
    println!("Hello");
    let mut env = PythonGymnasiumCartPoleState::new().unwrap();
    env.forward(SINGLE_PLAYER_ID, 1).unwrap();
    println!("world")
}