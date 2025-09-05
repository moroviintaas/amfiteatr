use crate::agent::*;
use crate::scheme::Scheme;


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
/// use amfiteatr_core::demo::{DemoScheme, DemoInfoSet};
/// let agents: Vec<Arc<Mutex<dyn ModelAgent<DemoScheme,(), DemoInfoSet >>>>;
/// //                      Scheme ----------------^
/// //                      Seed ------------------------^
/// //                      InformationSet-------------------------^
/// ```
/// This trait has blanket implementation for types implementing its supertraits
pub trait ModelAgent<S: Scheme, Seed, IS: InformationSet<S>>:

    AutomaticAgent<S>
    //SelfEvaluatingAgent<S, Assessment= <IS as EvaluatedInformationSet<S>>::RewardType>
    + ReseedAgent<S, Seed>
    + MultiEpisodeAutoAgent<S, Seed>
    + StatefulAgent<S, InfoSetType=IS>
    + TracingAgent<S, IS>
    + RewardedAgent<S>
    + Send
{}

impl<
    S: Scheme,
    Seed,
    IS: InformationSet<S>,
    T: AutomaticAgent<S>
        //+ SelfEvaluatingAgent<S, Assessment= <IS as EvaluatedInformationSet<S>>::RewardType>
        + ReseedAgent<S, Seed>
        + MultiEpisodeAutoAgent<S, Seed>
        + StatefulAgent<S, InfoSetType=IS>
        + TracingAgent<S, IS>
        + RewardedAgent<S>
        + Send

> ModelAgent<S, Seed, IS> for T {}



