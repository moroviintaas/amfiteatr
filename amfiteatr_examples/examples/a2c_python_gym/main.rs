use crate::env::{PythonGymnasiumCartPoleState, SINGLE_PLAYER_ID};
use amfiteatr_core::env::EnvironmentStateSequential;

mod env;

fn main() {
    println!("Hello");
    let mut env = PythonGymnasiumCartPoleState::new().unwrap();
    env.forward(SINGLE_PLAYER_ID, 1).unwrap();
    println!("world")
}