# Changelog

---
## Version 0.6.0
+ Added step legality validation note in `AgentTrajectory` and `AgentStepView`
+ Renamed `PolicyAgent::take_action` to `PolicyAgent::do_action`, as _take_ is conventionally used in Rust to move out field
+ Changed trait `TracingAgent`, now does not demand functions `commit_trace` nor `finalize_trajectory`.
Only trajectory read and reset is needed, maintaining trajectory is now internal responsibility of agent (can and probably should be private logic of agent).

## Version 0.5.0
+ Rework of tracing - replaced `Trajectory` with `AgentTrajectory` without
need of information set to implement `EvaluatedInformationSet`;
and `EnvTrajectory` with `GameTrajectory` working in similar way like `AgentTrajectory`.
+ Changed trait `EvaluatedInformationSet` to have generic reward allowing multiple implementations for single information set.
+ Changed `ActingAgent` trait's functions to return `Result` wrapped outputs
+ Merged traits [`AutomaticAgent`] and [`AutomaticAgentRewarded`] into [`AutomaticAgent`].

## Version 0.4.0
+ Renamed `AmfiRLError` to `AmfiteatrRlError`.
+ Added `serde` support for protocol messages.
+ Made logs in `core` and `rl` crates optional, activated with feature.
+ Added required function `switch_explore for` `LearningNetworkPolicy` trait
+ Added `tch::VarStore` get access in `RlSimpleLearningAgent` to access network parameters.
+ Added minor fix for bug occurring when every trajectory was empty in implemented neural network policies (Q, A2C).
___
## Version 0.3.0
+ Accumulated crates previously stored in submodules in single repository.
+ Updated `LeariningNetworkPolicy` implementation for A2C (`ActorCriticPolicy`).
+ Renamed trait `ConvertToTensor` to `CtxTryIntoTensor`.
+ Renamed trait `TryConvertFromTensor` to `CtxTryFromTensor`
+ Removed traits `EpisodeMemoryAgent`, `MultiEpisodeAutoAgent` and `MultiEpisodeAutoAgentRewarded`.
+ Added trait `MultiEpisodeAutoAgent` replacing `EpisodeMemoryAgent`, `MultiEpisodeAutoAgent` and `MultiEpisodeAutoAgentRewarded` .

Deprecated:  
- trait `ActionTensor` (suggested use of `CtxTryFromTensor` and `CtxTryIntoTensor`)


___