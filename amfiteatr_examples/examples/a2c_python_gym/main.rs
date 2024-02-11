use crate::env::PythonGymnasiumCartPoleState;

mod env;

fn main() {
    println!("Hello");
    let env = PythonGymnasiumCartPoleState::new().unwrap();
    println!("world")
}