//! # Communication
//! ## Communication models
//! Currently there are two models for communication between environment and agents
//! 1. Bidirectional channels between environment and every agent;
//! 2. Single environment port to communicate with any agent
//! ### 1. Bidirectional channels
//! For the game with _N_ players environment is equipped with _N_
//! [`endpoints`](crate::comm::BidirectionalEndpoint).
//! On th environment side endpoints should be stored in a way providing random
//! access to efficiently select endpoint to use.
//! Agents send [`AgentMessage`][crate::domain::AgentMessage] that is guaranteed to
//! reach environment, similarly environment send [`EnvironmentMessage`](crate::domain::EnvironmentMessage),
//! using previously selected endpoint.
//! ```notrust
//!  -------------------------------------------
//! |                Environment                |
//! |                 ---------------------     |    ---------
//! |                | Endpoint to agent 1 | ---+---| Agent 1 |
//! |                 ---------------------     |    ---------
//! |                 ---------------------     |    ---------
//! |                | Endpoint to agent 2 | ---+---| Agent 2 |
//! |                 ---------------------     |    ---------
//! |                          ...              |
//! |                 ---------------------     |    ---------
//! |                | Endpoint to agent N| ---+---| Agent N  |
//! |                 ---------------------     |    ---------
//!  -------------------------------------------
//! ```
//!
//! The keynote is: To receive on environment select endpoint and then receive message
//! (wrapped as Result) from this endpoint:
//! ``` no_run, compile_fail
//! let selected_player = your_method_to_select();
//! let mut endpoint = endpoints.get_mut(&selected_player);
//! //receiving
//! if let Some(message) = endpoint.receive_non_blocking().unwrap(){
//!     //do something
//! }
//! //sending
//! endpoint.send(some_message).unwrap();
//! ```
//!
//! Choice of api objects to use this model:
//! 1. [`StdAgentEndpoint`](crate::comm::StdAgentEndpoint)
//! 2. [`StdEnvironmentEndpoint`](crate::comm::StdEnvironmentEndpoint)
//! 3. [`HashMapEnvironment`](crate::env::HashMapEnvironment) or [`TracingHashMapEnvironment`](crate::env::TracingHashMapEnvironment)
//! 4. [`AgentGen`](crate::agent::AgentGen) or [`TracingAgentGen`](crate::agent::TracingAgentGen)
//! with communicating endpoint sending [`AgentMessage`](crate::domain::AgentMessage)
//!
//!
//!
//! __Advantages of this construction:__
//! 1. It is possible to use `Box<dyn BidirectionalEndpoint>` and have some agents
//! using standard [`channel`](std::sync::mpsc::channel) backend for local agents and some
//! network implemented endpoints (e.g. TCP).
//! 2. Intuitive concept. Environment can store endpoints in [`HashMap`](std::collections::HashMap)
//! keyed with agent id. When it wants to send message it just selects endpoint from the map
//! and uses it to send.
//! 3. You can select the order of reading from agent and avoid being spammed by
//! badly working agent.
//!
//!
//! __Disadvantages of this construction:__
//! 1. Some receiving strategy needs to be introduced, at any time
//! any agent can send message, e.g. with some error. With many endpoints
//! environment must constantly switch listening between them.
//! Switching may cost some processor time, and depends on channel efficiency
//! in determining if there is waiting message or not.

//! ### 2. Environment central communicator
//! In this concept we do not use separated bidirectional channels between
//! environment and agents. Environment use single communication adapter
//! to communicate with every agent.
//! It uses only one receiving interface which looks like queue of pairs
//!
//! To receive message you can use struct implementing [`EnvironmentAdapter`](crate::comm::EnvironmentAdapter)
//! You then receive pair of identifier ([`AgentIdentifier`](crate::agent::AgentIdentifier)) of sender and his
//! [`AgentMessage`](crate::domain::AgentMessage).
//! This may be natural choice if all agents are run locally.
//! Agents sending to environment can be easily implemented using
//! [`mpsc::channel`](std::sync::mpsc::channel).
//! Sending from environment to agents in this case is made with _N_ mpsc channels.
//!
//! ```notrust
//!  -------------------------------------------
//! |                Environment                |
//! |                ----------------------     |    ---------
//! |               |                      | ---+---| Agent 1 |
//! |               |                      |    |    ---------
//! |               |                      |    |    ---------
//! |               |  Environment Adapter | ---+---| Agent 2 |
//! |               |                      |    |    ---------
//! |               |                      |    |       ...
//! |               |                      |    |    ---------
//! |               |                      | ---+---| Agent N |
//! |                ----------------------     |    ---------
//!  -------------------------------------------
//! ```
//! //! Choice of api objects to use this model:
//! 1. [`AgentMpscPort`](crate::comm::AgentMpscAdapter)
//! 2. [`EnvironmentMpscPort`](crate::comm::EnvironmentMpscPort)
//! 3. [`BasicEnvironment`](crate::env::HashMapEnvironment) or [`TracingEnvironment`](crate::env::TracingHashMapEnvironment)
//! 4. [`AgentGen`](crate::agent::AgentGen) or [`TracingAgentGen`](crate::agent::TracingAgentGen)
//!
//! ``` no_run, compile_fail
//! // we do not specify agent
//! let mut endpoint = borrow_some_endpoint();
//! //receiving
//! // agent_id is delivered by endpoint (endpoint recognises which agent sent this message)
//! if let Some((agent_id, message)) = endpoint.receive_non_blocking().unwrap(){
//!     //do something
//!
//! }
//! //sending
//! endpoint.send(some_message).unwrap();
//! ```
//! Such adapter may be made using HashMap of [`BidirectionalEndpoint`s](crate::comm::BidirectionalEndpoint)
//! __Advantages of this construction:__
//! 1. In models with all agents local it is efficiently implemented using standard mpsc
//! channels
//! 2. Environment does not need to implement switching for incoming traffic
//! (however if switching is needed, adapter must do this)
//! __Disadvantages of this construction:__
//! 1. It may be difficult to combine different communication backends.
//! When some agents connects with mpsc and some with network protocols endpoint
//! must switch receiving over these two methods and must know which agent is connected
//! via which backend.
//!
//! __Note__
//! It is possible to wrap first model endpoints in second model's adapters.
//! On the environment's side endpoints are collected in HashMap
//! (or Vec if Agents can be converted to subsequent usize indexes).
//! Sending is made by selecting proper endpoint and using it to send message.
//! For receiving it is needed to select switching strategy (for example round-robin)
//! and receiving message on any endpoint then returning this message with
//! agent_id attached to used endpoint.
mod endpoint;
mod std_channel;

pub use std_channel::*;
pub use endpoint::*;

