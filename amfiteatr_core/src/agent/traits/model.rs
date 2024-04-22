use crate::agent::*;
use crate::domain::DomainParameters;


/// This is one of proposed top level traits to use in dynamic structs like
/// [`Box<>`](std::boxed::Box) or [`Mutex<>`](std::sync::Mutex).
/// This is done because Mutex<dyn ...> can use only one non-auto trait so if you need dynamic agent
/// that can track his trace and can [`reseed`](ReseedAgent) it's information set before episode you probably want to use this or build something over it.
/// The reason why this trait needs specifying information set is that it is part of tracing interface
/// which operates on [`AgentStepView`].
/// If you don't need access to tracing and want to avoid providing concrete info set you could
/// probably use [`MultiEpisodeAutoAgent`].
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
pub trait ModelAgent<DP: DomainParameters, Seed, IS: InformationSet<DP>>:

    AutomaticAgent<DP>
    //SelfEvaluatingAgent<DP, Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
    + ReseedAgent<DP, Seed>
    + MultiEpisodeAutoAgent<DP, Seed>
    + StatefulAgent<DP, InfoSetType=IS>
    + TracingAgent<DP, IS>
    + RewardedAgent<DP>
    + Send
{}

impl<
    DP: DomainParameters,
    Seed,
    IS: InformationSet<DP>,
    T: AutomaticAgent<DP>
        //+ SelfEvaluatingAgent<DP, Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
        + ReseedAgent<DP, Seed>
        + MultiEpisodeAutoAgent<DP, Seed>
        + StatefulAgent<DP, InfoSetType=IS>
        + TracingAgent<DP, IS>
        + RewardedAgent<DP>
        + Send

> ModelAgent<DP, Seed, IS> for T {}