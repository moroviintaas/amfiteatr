use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock};
use rand::seq::IteratorRandom;
use crate::agent::info_set::InformationSet;
use crate::agent::{PresentPossibleActions};
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;

/// Trait meant for structures working as action selectors. Policy based on information set
/// must select one action if possible.
pub trait Policy<S: Scheme>: Send{
    /// Information set which for which this policy is meant to work.
    type InfoSetType: InformationSet<S>;

    /// Selects action based on information set.
    /// If at least one action is possible result should be `Ok(action)` otherwise `Err`.
    /// In special case where no action is possible or from the agent's point of view game is finished
    /// suggested error is [`AmfiteatrError::NoActionAvailable`](AmfiteatrError::NoActionAvailable).
    ///
    /// Migration from previous version: use `ok_or`
    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>>;

    /// This method is meant to be called at the beginning of the episode.
    /// It is not crucial for game protocol, yet you can use it to set something in policy for the episode.
    /// For example select one variant of mixed policy or entity from genetic population.
    fn call_on_episode_start(&mut self) -> Result<(), AmfiteatrError<S>>{
        #[cfg(feature = "log_trace")]
        log::trace!("Initial setting of policy.");
        Ok(())
    }

    /// This method is meant to be called at the end of the episode.
    /// It is not crucial for game protocol, yet you can use it to set something in policy after the episode.
    /// For example note something about policy and performance of choice at the beginning of the episode.
    /// That might be useful in constructing some genetic policies, that try different variants for episodes autonomously,
    /// and they internally rank their tested variants later.
    #[allow(unused_variables)]
    fn call_on_episode_finish(&mut self,
                              final_env_payoff: S::UniversalReward,
    ) -> Result<(), AmfiteatrError<S>>
    {
        #[cfg(feature = "log_trace")]
        log::trace!("End of episode setting of policy.");
        Ok(())
    }

    /// This method is meant to be called between epochs.
    /// It is not crucial for game protocol, yet you can use it to set something in policy between epochs
    /// For example clean episode notation.
    fn call_between_epochs(&mut self)  -> Result<(), AmfiteatrError<S>>{
        #[cfg(feature = "log_trace")]
        log::trace!("Between epochs setting of policy.");
        Ok(())
    }


}

impl<S: Scheme, P: Policy<S>> Policy<S> for Arc<Mutex<P>>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {

        match self.as_ref().lock(){
            Ok(internal_policy) => {
                internal_policy.select_action(state)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_on_episode_start(&mut self) -> Result<(), AmfiteatrError<S>> {
        match self.as_ref().lock(){
            Ok(mut internal_policy) => {
                internal_policy.call_on_episode_start()
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_on_episode_finish(&mut self, final_env_reward: S::UniversalReward,) -> Result<(), AmfiteatrError<S>> {
        match self.as_ref().lock(){
            Ok(mut internal_policy) => {
                internal_policy.call_on_episode_finish(final_env_reward)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_between_epochs(&mut self)  -> Result<(), AmfiteatrError<S>>{
        match self.as_ref().lock(){
            Ok(mut internal_policy) => {
                internal_policy.call_between_epochs()
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }
}

impl<S: Scheme, P: Policy<S>> Policy<S> for Mutex<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {

        match self.lock(){
            Ok(internal_policy) => {
                internal_policy.select_action(state)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_on_episode_start(&mut self) -> Result<(), AmfiteatrError<S>> {
        match self.lock(){
            Ok(mut internal_policy) => {
                internal_policy.call_on_episode_start()
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_on_episode_finish(&mut self, final_env_reward: S::UniversalReward,) -> Result<(), AmfiteatrError<S>> {
        match self.lock(){
            Ok(mut internal_policy) => {
                internal_policy.call_on_episode_finish(final_env_reward)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_between_epochs(&mut self)  -> Result<(), AmfiteatrError<S>>{
        match self.lock(){
            Ok(mut internal_policy) => {
                internal_policy.call_between_epochs()
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }
}
impl<S: Scheme, P: Policy<S>> Policy<S> for RwLock<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {

        match self.read(){
            Ok(internal_policy) => {
                internal_policy.select_action(state)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_on_episode_start(&mut self) -> Result<(), AmfiteatrError<S>> {
        match self.write(){
            Ok(mut internal_policy) => {
                internal_policy.call_on_episode_start()
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_on_episode_finish(&mut self, final_env_reward: S::UniversalReward,) -> Result<(), AmfiteatrError<S>> {
        match self.write(){
            Ok(mut internal_policy) => {
                internal_policy.call_on_episode_finish(final_env_reward)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }

    fn call_between_epochs(&mut self)  -> Result<(), AmfiteatrError<S>>{
        match self.write(){
            Ok(mut internal_policy) => {
                internal_policy.call_between_epochs()
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }
}


/// Generic random policy - selects action at random based on iterator of possible actions
/// provided by [`InformationSet`](crate::agent::InformationSet).
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct RandomPolicy<S: Scheme, State: InformationSet<S>>{
    state: PhantomData<State>,
    _spec: PhantomData<S>
}


impl<S: Scheme, InfoSet: InformationSet<S>> RandomPolicy<S, InfoSet>{
    pub fn new() -> Self{
        Self{state: PhantomData, _spec: PhantomData}
    }
}



impl<S: Scheme, InfoSet: PresentPossibleActions<S>> Policy<S> for RandomPolicy<S, InfoSet>
where <<InfoSet as PresentPossibleActions<S>>::ActionIteratorType as IntoIterator>::IntoIter : ExactSizeIterator{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        let mut rng = rand::rng();
        state.available_actions().into_iter().choose(&mut rng).ok_or_else(|| AmfiteatrError::NoActionAvailable {
            context: "Random policy".into()})
    }
}

impl<S: Scheme, P: Policy<S>> Policy<S> for Box<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        self.as_ref().select_action(state)
    }
}


#[cfg(feature = "mcp")]
mod mcp{
    use std::marker::PhantomData;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use crate::agent::{InformationSet, Policy};
    use crate::scheme::Scheme;
    use std::sync::Arc;
    use rmcp::handler::server::wrapper::Parameters;
    use tokio::sync::Mutex;
    use rmcp::ErrorData;
    use rmcp::model::{CallToolResult, Content};
    use std::default::Default;
    use crate::util::mcp::McpReqReward;

    #[derive(Clone)]
    pub struct McpRequestSelectAction<SC: Scheme, IS: InformationSet<SC>>
    where
        IS: Serialize + for<'a> Deserialize<'a> + JsonSchema
    {
        pub information_set: IS,
        _sc: PhantomData<SC>,
    }


    /// Wraps policy structure by `Arc<tokio::sync::Mutex<P>>` and informational values.
    /// To be used as internal logic for policy MCP server.
    /// It uses `Arc<Mutex<>>` construction, because policy could have methods mutating its internal state.

    pub struct McpCorePolicy<SC: Scheme, IS: InformationSet<SC>, P: Policy<SC>>
    where
        IS: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        P: Policy<SC,  InfoSetType = IS>
    {
        internal: Arc<Mutex<P>>,
        policy_name: String,
        usage: String,
        _sc: PhantomData<SC>,
        _is: PhantomData<SC>,

    }

    impl <SC: Scheme, IS: InformationSet<SC>, P: Policy<SC>> McpCorePolicy<SC, IS, P>
    where
        IS: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        //SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        <SC as Scheme>::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        <SC as Scheme>::UniversalReward: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        P: Policy<SC,  InfoSetType = IS>
    {
        pub fn new(policy: P, policy_name: String, usage: String) -> Self{
            Self{
                internal: Arc::new(Mutex::new(policy)),
                policy_name, usage,
                _sc: Default::default(), _is: Default::default(),
            }
        }

        pub async fn select_action(&self, Parameters(McpRequestSelectAction{information_set, ..}): Parameters<McpRequestSelectAction<SC, IS>>)
        -> Result<CallToolResult, ErrorData>
        {
            let internal = self.internal.lock().await;

            match internal.select_action(&information_set) {
                Ok(action) => Ok(CallToolResult::success(vec![Content::json(action)?])),
                Err(e) => Err(ErrorData::internal_error(
                    format!("Failed to resolve action ({e})"),
                    Some(serde_json::to_value(&information_set).map_err(|e|{
                        ErrorData::internal_error(
                            format!("Failed to resolve action ({e}) and to serialize information set: {information_set:?}."),
                            None
                        )
                    })?))),
            }

        }

        pub async fn call_on_episode_start(&self)
                                   -> Result<CallToolResult, ErrorData>
        {
            let mut internal = self.internal.lock().await;

            match internal.call_on_episode_start(){
                Ok(_) => Ok(CallToolResult::success(vec![])),
                Err(e) => Err(ErrorData::internal_error(
                    format!("Failed to policy preparation on beginning of the episode ({e})"),
                    None
                ))
            }



        }

        pub async fn call_on_episode_finish(
            &self,
            Parameters(McpReqReward{reward}): Parameters<McpReqReward<SC>>
        )
                                           -> Result<CallToolResult, ErrorData>
        {
            let mut internal = self.internal.lock().await;

            match internal.call_on_episode_finish(reward){
                Ok(_) => Ok(CallToolResult::success(vec![])),
                Err(e) => Err(ErrorData::internal_error(
                    format!("Failed to policy preparation on beginning of the episode ({e})"),
                    None
                ))
            }



        }

        pub async fn call_between_epochs(&self)
                                           -> Result<CallToolResult, ErrorData>
        {
            let mut internal = self.internal.lock().await;

            match internal.call_between_epochs(){
                Ok(_) => Ok(CallToolResult::success(vec![])),
                Err(e) => Err(ErrorData::internal_error(
                    format!("Failed to policy preparation on beginning of the episode ({e})"),
                    None
                ))
            }



        }
    }
}


#[cfg(feature = "mcp")]
pub use mcp::*;



