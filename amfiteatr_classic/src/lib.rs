//! This crate provides minimal infrastructure to deal with classical game theory problems such as:
//! 1. [Prisoners' dilemma](https://en.wikipedia.org/wiki/Prisoner's_dilemma) (or any other game represented in 2x2 grid)
//! 2. [Replicator dynamic](https://en.wikipedia.org/wiki/Evolutionary_game_theory) problem based on games represented in 2x2 grid.

/// Module for agent related structs and traits.
pub mod agent;
/// Module for central environment related structs and traits.
pub mod env;
/// Module for definition of classic game scheme
pub mod domain;
/// Module for classic policies definitions
pub mod policy;

mod common;

pub use common::*;
