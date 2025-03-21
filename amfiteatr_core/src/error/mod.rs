//! # Errors
//! In this module there is hierarchical Error structure defined.
//! Top level error is [`AmfiteatrError`](crate::error::AmfiteatrError).
//! As the crates develops new error cases occurs and some may be merged or moved to different categories.
//! **As for now the error structure is very likely to change in the future.**
//!
//!
//! ## Why not `Anyhow`?
//! I would be very happy to use `anyhow` crate to deal with errors, however this model
//! clones errors when propagating error to other agents. Agents `send` errors to environment and use
//! second instance in their thread. Environment broadcasts error to all agents.
//!
//! Maybe in the future some structure like `Arc<_: Error> would be used. This will need some benchmarks also to be done.
//! One day maybe.
//!

mod comm;
mod amfiteatr;
mod protocol;
mod internal_error;
//mod setup;
mod convert;
mod model;
mod trajectory;
mod tensor;
mod data;

pub use comm::*;
pub use self::amfiteatr::*;
pub use protocol::*;
pub use internal_error::*;
//pub use setup::*;
pub use convert::*;
pub use model::*;
pub use trajectory::*;
pub use tensor::*;
pub use data::*;