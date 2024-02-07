use amfiteatr_core::agent::*;
use amfiteatr_core::domain::DomainParameters;
use crate::policy::LearningNetworkPolicy;


/// Trait representing agent that run automatically (with reward collection) (it does not
/// require any compatibility). This trait is not object safe because Policy traits has generic type
/// information set with generic parameter of [`DomainParameters`](amfiteatr_core::domain::DomainParameters).
///
pub trait NetworkLearningAgent<DP: DomainParameters>:
    AutomaticAgentRewarded<DP>
    + PolicyAgent<DP>
    + TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>
    where  <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
    <Self as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{
}

impl <DP: DomainParameters, T: AutomaticAgentRewarded<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>>
NetworkLearningAgent<DP> for T
where <T as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
<T as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{
}

/*
pub trait TestingAgent<DP: DomainParameters>: AutomaticAgent<DP>  + PolicyAgent<DP>
 + TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>
where <Self as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>{}

impl <DP: DomainParameters, T: AutomaticAgent<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>>

TestingAgent<DP> for T
where <T as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{}

 */


/// Trait representing agent that run automatically (with reward collection) and cam be reseeded
/// for subsequent game episodes.
/// For now this trait requires both environment sources reward and agent self provided assessment.
/// If you only want to define one you can set not needed to by of type [`NoneReward`](amfiteatr_core::domain::NoneReward).
/// This trait is object safe, however collections of dynamically typed agents of this trait must
/// share the same type of information set, because [`LearningNetworkPolicy`](crate::policy::LearningNetworkPolicy)
/// uses trajectory including information set.
pub trait RlModelAgent<DP: DomainParameters, Seed, IS: EvaluatedInformationSet<DP>>:
    AutomaticAgentRewarded<DP>
    + SelfEvaluatingAgent<DP,  Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
    + MultiEpisodeAutoAgentRewarded<DP, Seed>
    + PolicyAgent<DP> + StatefulAgent<DP, InfoSetType=IS>
    + TracingAgent<DP, IS>
    + Send

where <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
{}




impl<
    DP: DomainParameters,
    Seed,
    IS: EvaluatedInformationSet<DP>,
    T: AutomaticAgentRewarded<DP>
        + SelfEvaluatingAgent<DP,  Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
        + MultiEpisodeAutoAgentRewarded<DP, Seed>
        + PolicyAgent<DP> + StatefulAgent<DP, InfoSetType=IS>
        + TracingAgent<DP, IS>
        + Send

> RlModelAgent<DP, Seed, IS> for T
where <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,{

}

