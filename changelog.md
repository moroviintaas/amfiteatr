# Changelog

## Version 0.7.0
+ Changed `select_action` method from `Policy` trait signature. From returning `Option<Action>` it now forces to return `Result<Action, AmfiteatrError<_>>`.
The change is made because some policy algorithms may actually fail.
Especially encountered during implementation of neural network policies.
This obviously breaks code. For use case when previously policy returned `Option::None`. because no action was possible - 
the suggested quick fix is transforming `Option` to proper `Result` by
```
.ok_or_else(|| AmfiteatrError::NoActionAvailable {
       context: "Random policy".into()})
}
```
+ Added special error variant `AmfiteatrError::NoActionAvailable` to indicate that agents believes that no action can be played.
+ Added new variants for `AmfiteatrError` (`Tensor` and `Data`), while removing `DataConvert`.
+ Renamed traits `CtxTryFromTensor`, `CtxTryToTensor`, `CtxTryFromMultipleTensors`, `CtxTryIntoMultipleTensors` to more verbose names `ContextTryFromTensor`, `ContextTryToTensor`, `ContextTryFromMultipleTensors`, `ContextTryIntoMultipleTensors`.
+ Changing `ContextTryConvert...` traits that functions return `Result<_,ConvertError` (from `AmfiteatrError`) instead of `TensorRepresentation` from `AmfiteatrRlError`
---
## Version 0.6.0
+ Added step legality validation note in `AgentTrajectory` and `AgentStepView`
+ Renamed `PolicyAgent::take_action` to `PolicyAgent::select_action`, as _take_ is conventionally used in Rust to move out field
+ Changed trait `TracingAgent`, now does not demand functions `commit_trace` nor `finalize_trajectory`.
Only trajectory read and reset is needed, maintaining trajectory is now internal responsibility of agent (can and probably should be private logic of agent)
+ Changed trait `CommunicatingAgent`, now without associated type `CommunicationError`, error type is now fixed.
+ Renamed `EnvironmentStateSequential` trait to `SequentialGameState` and `EnvironmentStateUniScore` to `GameStateWithPayoffs`
+ Added traits for shaping network having actor selecting from multi-parameter, e.g `TensorCriticMultiActor`rfgk 
+ Removed generic implementation `AutoEnvironment<_> for E`, but added `AutoEnvironment<_> for BasicEnvironment<_>`,
previously it would block custom implementation. In the future I want to provide derive macro implementation to choose
if one want to use provided implementation. Similar with `AutoEnvironmentWithScores` and `AutoEnvironmentWithScoresAndPenalties`.
+ Updated `LearningNetworkPolicy` trait, to not require specification of config and network. Removed getters method for them from network.
+ Added **Experimental** traits `ProcedureAgent`, `CliAgent`, `AssistingPolicy` and  simple interactive agent with trait `CliAgent` using `TurnCommand`.
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