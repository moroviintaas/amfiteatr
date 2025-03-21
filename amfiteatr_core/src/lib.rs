//! # amfiteatr
//!
//!
//! Crate providing framework to build models simulating game theory problems with support for machine learning techniques.
//! It is designed to help model problems involving many players.
//!
//! ## Examples
//! For examples look at [`amfiteatr_examples`](https://github.com/moroviintaas/amfiteatr_examples.git)
//! ## State
//! The crate is in research/experimental state. API is dynamically changing on every release.
//! U should probably avoid using it in production.
//! ## Licence: MIT

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
pub mod util;

//use amfiteatr_proc_macro as macros;

pub mod reexport{
    pub use nom;
    pub use amfiteatr_proc_macro::*;
}



//pub mod world;

//mod map;


//pub use map::*;