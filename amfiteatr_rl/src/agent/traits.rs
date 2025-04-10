use std::iter::Sum;
use tch::nn::VarStore;
use amfiteatr_core::agent::*;
use amfiteatr_core::comm::BidirectionalEndpoint;
use amfiteatr_core::domain::{AgentMessage, DomainParameters, EnvironmentMessage, Renew};
use amfiteatr_core::error::{CommunicationError};
use crate::error::AmfiteatrRlError;
use crate::policy::{LearnSummary, LearningNetworkPolicy};
use crate::tensor_data::FloatTensorReward;


/// Trait representing agent that run automatically (with reward collection) (it does not
/// require any compatibility). This trait is not object safe because Policy traits has generic type
/// information set with generic parameter of [`DomainParameters`](amfiteatr_core::domain::DomainParameters).
///
pub trait NetworkLearningAgent<DP: DomainParameters>:
    AutomaticAgent<DP>
    + PolicyAgent<DP>
    + TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>
    where  <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
    <Self as StatefulAgent<DP>>::InfoSetType: InformationSet<DP>
{
}

impl <DP: DomainParameters, T: AutomaticAgent<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>>
NetworkLearningAgent<DP> for T
where <T as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
<T as StatefulAgent<DP>>::InfoSetType: InformationSet<DP>
{
}




/// Trait representing agent that run automatically (with reward collection) and cam be reseeded
/// for subsequent game episodes.
/// For now this trait requires both environment sources reward and agent self provided assessment.
/// If you only want to define one you can set not needed to by of type [`NoneReward`](amfiteatr_core::domain::NoneReward).
/// This trait is object safe, however collections of dynamically typed agents of this trait must
/// share the same type of information set, because [`LearningNetworkPolicy`](crate::policy::LearningNetworkPolicy)
/// uses trajectory including information set.
pub trait RlModelAgent<DP: DomainParameters, Seed, IS: InformationSet<DP>>:
    AutomaticAgent<DP>
    //+ SelfEvaluatingAgent<DP,  Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
    + ReseedAgent<DP, Seed>
    + PolicyAgent<DP> + StatefulAgent<DP, InfoSetType=IS>
    + MultiEpisodeTracingAgent<DP, IS, Seed>
    + RewardedAgent<DP>
    + Send

where <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
{}




impl<
    DP: DomainParameters,
    Seed,
    IS: InformationSet<DP>,
    T: AutomaticAgent<DP>
        //+ SelfEvaluatingAgent<DP,  Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
        + ReseedAgent<DP, Seed>
        + PolicyAgent<DP> + StatefulAgent<DP, InfoSetType=IS>
        + MultiEpisodeTracingAgent<DP, IS, Seed>
        + RewardedAgent<DP>
        + Send

> RlModelAgent<DP, Seed, IS> for T
where <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,{

}


pub trait RlSimpleTestAgent<DP: DomainParameters, Seed>:
AutomaticAgent<DP> + ReseedAgent<DP, Seed> + Send{

}

pub trait RlSimpleLearningAgent<DP: DomainParameters, Seed, LS: Send>:
AutomaticAgent<DP> + ReseedAgent<DP, Seed> + Send + MultiEpisodeAutoAgent<DP, Seed>
{
    fn simple_apply_experience(&mut self) -> Result<LS, AmfiteatrRlError<DP>>;
    //fn clear_experience(&mut self) -> Result<(), AmfiteatrError<DP>>;

    fn set_exploration(&mut self, explore: bool);

    fn get_var_store(&self) -> &VarStore;

}




impl<
    DP: DomainParameters,
    Seed,
    P: LearningNetworkPolicy<DP, Summary = LS >,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>> + Send,
    LS: Send,
>

RlSimpleLearningAgent<DP, Seed, LS> for TracingAgentGen<DP, P, Comm, >
    where <P as Policy<DP>>::InfoSetType: InformationSet<DP> + Renew<DP, Seed>,
          <DP as DomainParameters>::UniversalReward: FloatTensorReward,
    Self: AutomaticAgent<DP> + MultiEpisodeAutoAgent<DP, Seed> + PolicyAgent<DP, Policy=P>
    {
    fn simple_apply_experience(&mut self) -> Result<LS, AmfiteatrRlError<DP>> {
        let episodes = self.take_episodes();

        self.policy_mut().train_on_trajectories_env_reward(&episodes)
    }

    fn set_exploration(&mut self, explore: bool) {
        self.policy_mut().switch_explore(explore)
    }
    fn get_var_store(&self) -> &VarStore{
        self.policy().var_store()
    }


}

