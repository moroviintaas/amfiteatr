use amfiteatr_core::agent::*;
use amfiteatr_core::comm::BidirectionalEndpoint;
use amfiteatr_core::scheme::{AgentMessage, Scheme, EnvironmentMessage, Renew};
use amfiteatr_core::error::{CommunicationError};
use amfiteatr_core::util::TensorboardSupport;
use crate::error::AmfiteatrRlError;
use crate::policy::LearningNetworkPolicyGeneric;
use crate::tensor_data::FloatTensorReward;


/// Trait representing agent that run automatically (with reward collection) (it does not
/// require any compatibility). This trait is not object safe because Policy traits has generic type
/// information set with generic parameter of [`DomainParameters`](amfiteatr_core::scheme::Scheme).
///
pub trait NetworkLearningAgent<S: Scheme>:
    AutomaticAgent<S>
    + PolicyAgent<S>
    + TracingAgent<S, <Self as StatefulAgent<S>>::InfoSetType>
    where  <Self as PolicyAgent<S>>::Policy: LearningNetworkPolicyGeneric<S>,
    <Self as StatefulAgent<S>>::InfoSetType: InformationSet<S>
{
}

impl <S: Scheme, T: AutomaticAgent<S>  + PolicyAgent<S>
+ TracingAgent<S, <Self as StatefulAgent<S>>::InfoSetType>>
NetworkLearningAgent<S> for T
where <T as PolicyAgent<S>>::Policy: LearningNetworkPolicyGeneric<S>,
<T as StatefulAgent<S>>::InfoSetType: InformationSet<S>
{
}




/// Trait representing agent that run automatically (with reward collection) and cam be reseeded
/// for subsequent game episodes.
/// For now this trait requires both environment sources reward and agent self provided assessment.
/// If you only want to define one you can set not needed to by of type [`NoneReward`](amfiteatr_core::scheme::NoneReward).
/// This trait is object safe, however collections of dynamically typed agents of this trait must
/// share the same type of information set, because [`LearningNetworkPolicy`](crate::policy::LearningNetworkPolicyGeneric)
/// uses trajectory including information set.
pub trait RlModelAgent<S: Scheme, Seed, IS: InformationSet<S>>:
    AutomaticAgent<S>
    //+ SelfEvaluatingAgent<S,  Assessment= <IS as EvaluatedInformationSet<S>>::RewardType>
    + ReseedAgent<S, Seed>
    + PolicyAgent<S> + StatefulAgent<S, InfoSetType=IS>
    + MultiEpisodeTracingAgent<S, IS, Seed>
    + RewardedAgent<S>
    + Send

where <Self as PolicyAgent<S>>::Policy: LearningNetworkPolicyGeneric<S>,
{}




impl<
    S: Scheme,
    Seed,
    IS: InformationSet<S>,
    T: AutomaticAgent<S>
        //+ SelfEvaluatingAgent<S,  Assessment= <IS as EvaluatedInformationSet<S>>::RewardType>
        + ReseedAgent<S, Seed>
        + PolicyAgent<S> + StatefulAgent<S, InfoSetType=IS>
        + MultiEpisodeTracingAgent<S, IS, Seed>
        + RewardedAgent<S>
        + Send

> RlModelAgent<S, Seed, IS> for T
where <Self as PolicyAgent<S>>::Policy: LearningNetworkPolicyGeneric<S>,{

}


pub trait RlSimpleTestAgent<S: Scheme, Seed>:
AutomaticAgent<S> + ReseedAgent<S, Seed> + Send{

}

pub trait RlSimpleLearningAgent<S: Scheme, Seed, LS: Send>:
AutomaticAgent<S> + ReseedAgent<S, Seed> + Send + MultiEpisodeAutoAgent<S, Seed>
{
    fn simple_apply_experience(&mut self) -> Result<LS, AmfiteatrRlError<S>>;
    //fn clear_experience(&mut self) -> Result<(), AmfiteatrError<S>>;

    fn set_exploration(&mut self, explore: bool);

    //fn get_var_store(&self) -> &VarStore;

}







impl<
    S: Scheme,
    Seed,
    P: LearningNetworkPolicyGeneric<S, Summary = LS >,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>> + Send,
    LS: Send,
>

RlSimpleLearningAgent<S, Seed, LS> for TracingAgentGen<S, P, Comm, >
    where <P as Policy<S>>::InfoSetType: InformationSet<S> + Renew<S, Seed>,
          <S as Scheme>::UniversalReward: FloatTensorReward,
    Self: AutomaticAgent<S> + MultiEpisodeAutoAgent<S, Seed> + PolicyAgent<S, Policy=P>
    {
    fn simple_apply_experience(&mut self) -> Result<LS, AmfiteatrRlError<S>> {
        let episodes = self.take_episodes();

        self.policy_mut().train(&episodes)
    }

    fn set_exploration(&mut self, explore: bool) {
        self.policy_mut().switch_explore(explore)
    }

        /*
    fn get_var_store(&self) -> &VarStore{
        self.policy().var_store()
    }

         */


}

