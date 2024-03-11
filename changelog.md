# Changelog
___
## Version 0.3.0
+ Accumulated crates previously stored in submodules in single repository.
+ Updated `LeariningNetworkPolicy` implementation for A2C (`ActorCriticPolicy`)
+ Renamed trait `ConvertToTensor` to `CtxTryIntoTensor`
+ Renamed trait `TryConvertFromTensor` to `CtxTryFromTensor`
+ Removed traits `EpisodeMemoryAgent`, `MultiEpisodeAutoAgent` and `MultiEpisodeAutoAgentRewarded`
+ Added trait `MultiEpisodeAutoAgent` replacing `EpisodeMemoryAgent`, `MultiEpisodeAutoAgent` and `MultiEpisodeAutoAgentRewarded` 

Deprecated:  
- trait `ActionTensor` (suggested use of `CtxTryFromTensor` and `CtxTryIntoTensor`)


___