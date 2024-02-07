use crate::agent::*;
use crate::domain::DomainParameters;


/// This is one of proposed top level traits to use in dynamic structs like
/// [`Box<>`](std::boxed::Box) or [`Mutex<>`](std::sync::Mutex).
/// This is done because Mutex<dyn ...> can use only one non-auto trait so if you need dynamic agent
/// that can track his trace and can [`reseed`](ReseedAgent) it's information set before episode you probably want to use this or build something over it.
/// The reason why this trait needs specifying information set is that it is part of tracing interface
/// which operates on [`AgentTraceStep<...>`](AgentTraceStep).
/// If you don't need access to tracing and want to avoid providing concrete info set you could
/// probably use [`MultiEpisodeAutoAgent`](EpisodeMemoryAutoAgent) or [`MultiEpisodeAutoAgentRewarded`](MultiEpisodeAutoAgentRewarded).
/// ```
/// use std::sync::{Arc, Mutex};
/// use amfiteatr_core::agent::ModelAgent;
/// use amfiteatr_core::demo::{DemoDomain, DemoInfoSet};
/// let agents: Vec<Arc<Mutex<dyn ModelAgent<DemoDomain,(), DemoInfoSet >>>>;
/// //                      Domain----------------^
/// //                      Seed ------------------------^
/// //                      InformationSet-------------------------^
/// ```
/// This trait has blanket implementation for types implementing it's supertraits
pub trait ModelAgent<DP: DomainParameters, Seed, IS: EvaluatedInformationSet<DP>>:

    AutomaticAgentRewarded<DP> +
    SelfEvaluatingAgent<DP, Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
    + MultiEpisodeAutoAgentRewarded<DP, Seed>
    //+ PolicyAgent<DP>
    + StatefulAgent<DP, InfoSetType=IS>
    + TracingAgent<DP, IS>
    + Send
{}

impl<
    DP: DomainParameters,
    Seed,
    IS: EvaluatedInformationSet<DP>,
    T: AutomaticAgentRewarded<DP>
        + SelfEvaluatingAgent<DP, Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
        + MultiEpisodeAutoAgentRewarded<DP, Seed>
        //+ PolicyAgent<DP>
        + StatefulAgent<DP, InfoSetType=IS>
        + TracingAgent<DP, IS>
        + Send

> ModelAgent<DP, Seed, IS> for T {}