//! # amfi
//!
//! __Licence:__ MIT
//!
//! Crate providing framework to build models simulating game theory problems.
//! It is designed to help model problems involving many players.
//! For examples look at [`amfiteatr_examples`](https://github.com/moroviintaas/amfiteatr_examples.git)


/// Traits and generic implementations of agent (player).
pub mod agent;
/// Generic structs used in communication between _agents_ and _environment_.
pub mod domain;
/// Traits and basic implementation for communication driving structs.
pub mod comm;
/// Structures used for error handling in framework.
pub mod error;
/// Traits and generic implementations for game controlling environment.
pub mod env;
/// Module with demonstration constructions
pub mod demo;

//pub mod world;

//mod map;


//pub use map::*;