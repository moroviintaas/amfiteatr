# Changelog

## Version 0.12.0
+ (**super breaking**) - renamed `domain::DomainParameters` to `scheme::Scheme`, because domain parameters are inappropriate here (it is used in elliptic curve cryptography).
This breaks almost everything, sorry. To migrate please refactor all occurrences.
+ Introduced experimental mutable calls on policy, that are called at the beginning, the end of episodes and between epochs.
+ Renamed `LearningNetworkPolicy<S: DomainParameters>` with generic `Summary` type to `LearningNetworkPolicyGeneric`.
Introduced new `LearningNetworkPolicy<S: DomainParameters> LearningNetworkPolicyGeneric<S>` with `Summary` fixed on `LearnSummary`  type.
+ In `RoundRobinEnvironment` changed names of `run_round_robin` and `run_round_robin_truncating` to
respectively `run_round_robin_no_rewards` and `run_round_robin_no_rewards_truncating`.
Making it more explicit to call environments without providing rewards.
Methods `run_round_robin` and `run_round_robin_truncating` are now parts of `RoundRobinUniversalEnvironment` trait.
Changed because running with rewards should probably be default behaviour, as it is more commonly used.
+ Renamed methods `LearningNetworkPolicyGeneric::train_on_trajectories` to `LearningNetworkPolicyGeneric::train_generic` and
`LearningNetworkPolicyGeneric::train_on_trajectories_env_rewards` to `LearningNetworkPolicyGeneric::train`
+ Added method `first_observations` to `SequentialGameState` to be called on the beginning of the game 
to send agents initial observations (now they can be provided by environment, instead of on information set initialization).
This is logically necessary.
+ Added provided method `current_universal_score_set_without_commit` to trait `RewardedAgent` to allow not only adding reward fragment to uncommited step draft, but also
set uncommitted step draft's partial reward so that total payoff is has certain value. It is helpful for models without unrolling episode in protocol.

## Version 0.11.0
+ Changed masking in A2C and PPO policies to use `f_where_self` instead of `f_mul`,
now masks must be `Tensor` with datatype `Bool`

## Version 0.10.0
+ From trait `LearningNetworkPolicy` removed methods `var_store()` and `var_store_mut_methods()` because they were returning reference to internal `VarStore` which is specific to `tch` and what's more important it prevents from making `Arc<Mutex<impl LearningNetworkPolicy>>`
+ Added blanket implementation for `LearningNetworkPolicy` for `Arc<Mutex<impl LearningNetworkPolicy>>
## Version 0.9.0
+ Changes in traits `RoundRobinEnvironment` `RoundRobinEnvironmentUniversalEnvironment`
`RoundRobinPenalisingUniversalEnvironment` - added respectively `run_round_robin_truncating`, 
`run_round_robin_with_rewards_truncating` and `run_round_robin_with_rewards_penalise_truncating` 
that introduce truncating act certain number of steps. Previous methods are auto implemented using these new with `None` argument.

## Version 0.8.0
+ Added policies `PolicyDiscreteA2C`, `PolicyMaskingDiscreteA2C`, `PolicyMaskingMultiDiscreteA2C`
+ Added `tboard` writer support for `PolicyDiscretePPO` and `PolicyMultiDiscretePPO`;
+ Added traits `PolicyHelperA2C`, `PolicyTrainHelperA2C` and `PolicyTrainHelperPPO` for building neural network policies 
in a generic way.
+ Deprecated `PolicyHelperPPO`
+ Renamed `PolicyPpoDiscrete` to `PolicyDiscretePPO`, `PolicyMaskingPpoDiscrete` to `PolicyMaskingDiscretePPO`, `PolicyPpoMultiDiscrete` to `PolicyMultiDiscretePPO`, `PolicyMaskingPpoMultiDiscrete` to `PolicyMaskingMultiDiscretePPO` and `ConfigPpo` to `ConfigPPO`
+ Renamed `CommunicatingAdapterEnvironment` to `CommunicatingEnvironmentSingleQueue`,`BroadConnectedEnvironment` to `BroadcastingEnvironmentSingleQueue`.
+ Renamed `DirtyReseedEnvironment` to `ReseedEnvironmentWithObservation` and it's method `dirty_reseed` to `reseed_with_observation`.
+ Changed `NetworkLearningPolicy` trait to have associated type for summary of learning session.
Also changed methods `train_on_trajectories` and `train_on_trajectories_env_reward` to return `Result<Self::Summary,_>`.
+ Fixed GAE calculation in policy PPO.
+ Added `tensorboard` support in example "connect\_four" python baseline.
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
+ Renamed `ConversionFromTensor`, `ConversionToTensor`, `ConversionFromMultipleTensors`, `ConversionToMultipleTensors` into respectfully `TensorDecoding`, `TensorEncoding`, `MultiTensorDecoding`, `MultiTensorEncoding`.
+ Renamed traits `CtxTryFromTensor`, `CtxTryToTensor`, `CtxTryFromMultipleTensors`, `CtxTryIntoMultipleTensors` to names `ContextDecodeTensor`, `ContextEncodeTensor`, `ContextDecodeMultiTensor`, `ContextEncodeMultiTensor`.
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