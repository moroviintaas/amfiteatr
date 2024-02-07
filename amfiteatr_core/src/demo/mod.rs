//! # Minimal example
//! ```
//! use std::collections::HashMap;
//! use std::thread;
//! use amfiteatr_core::agent::{AgentGen, TracingAgentGen, AutomaticAgent, AutomaticAgentRewarded, RewardedAgent, RandomPolicy};
//! use amfiteatr_core::comm::StdEnvironmentEndpoint;
//! use amfiteatr_core::demo::{DemoInfoSet, DemoDomain, DemoState, DemoAgentID, DemoPolicySelectFirst};
//! use amfiteatr_core::env::*;
//!
//!
//! let bandits = vec![5.0, 11.5, 6.0];
//! let number_of_bandits = bandits.len();
//! let state = DemoState::new(bandits, 100);
//! let (comm_env_r, comm_agent_r) = StdEnvironmentEndpoint::new_pair();
//! let (comm_env_b, comm_agent_b) = StdEnvironmentEndpoint::new_pair();
//! let mut env_comms = HashMap::new();
//! env_comms.insert(DemoAgentID::Blue, comm_env_b);
//! env_comms.insert(DemoAgentID::Red, comm_env_r);
//! let mut environment = TracingHashMapEnvironment::new(state, env_comms);
//! let blue_info_set = DemoInfoSet::new(DemoAgentID::Blue, number_of_bandits);
//! let red_info_set = DemoInfoSet::new(DemoAgentID::Red, number_of_bandits);
//! let mut agent_blue = TracingAgentGen::new(blue_info_set, comm_agent_b, RandomPolicy::<DemoDomain, DemoInfoSet>::new());
//! let mut agent_red = AgentGen::new(red_info_set, comm_agent_r, DemoPolicySelectFirst{});
//!
//! thread::scope(|s|{
//!     s.spawn(||{
//!         environment.run_round_robin_with_rewards().unwrap();
//!     });
//!     s.spawn(||{
//!         agent_blue.run_rewarded().unwrap();
//!     });
//!     s.spawn(||{
//!         agent_red.run_rewarded().unwrap();
//!     });
//! });
//!
//! assert_eq!(environment.trajectory().list().len(), 200);
//! assert!(environment.actual_score_of_player(&DemoAgentID::Blue) > 10.0);
//! assert!(agent_blue.current_universal_score() > 10.0);
//! assert!(agent_red.current_universal_score() > 10.0);
//! assert!(agent_blue.current_universal_score() > agent_red.current_universal_score());
//! ```

use std::fmt::{Debug, Display, Formatter};
use rand::{thread_rng};
use rand::distributions::Uniform;
use crate::agent::{AgentIdentifier, Policy, PresentPossibleActions};
use crate::demo::DemoAgentID::{Blue, Red};
use crate::domain::{Action, DomainParameters, Renew};
use crate::env::{EnvironmentStateSequential, EnvironmentStateUniScore};
use rand::distributions::Distribution;
use crate::agent::{InformationSet, EvaluatedInformationSet};



#[derive(Clone, Debug)]
pub struct DemoAction(pub u8);
impl Display for DemoAction{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl Action for DemoAction{}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, )]
pub enum DemoAgentID{
    Blue,
    Red
}
impl Display for DemoAgentID{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}



impl AgentIdentifier for DemoAgentID{}

#[derive(Copy, Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub struct DemoError{}
impl Display for DemoError{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DemoError")
    }
}



#[derive(Clone, Debug)]
pub struct DemoDomain {}

impl DomainParameters for DemoDomain {
    type ActionType = DemoAction;
    type GameErrorType = DemoError;
    type UpdateType = (DemoAgentID, DemoAction, f32);
    type AgentId = DemoAgentID;
    type UniversalReward = f32;
}

#[derive(Clone, Debug)]
pub struct DemoState{
    ceilings: Vec<f32>,
    max_rounds: u32,
    rewards_red: Vec<f32>,
    rewards_blue: Vec<f32>,
}

impl DemoState{
    pub fn new(ceilings: Vec<f32>, max_rounds: u32) -> Self{
        Self{ceilings, max_rounds, rewards_red: Vec::default(), rewards_blue: Vec::default()}
    }
}
impl EnvironmentStateSequential<DemoDomain> for DemoState{
    type Updates = Vec<(DemoAgentID, (DemoAgentID, DemoAction, f32))>;

    fn current_player(&self) -> Option<DemoAgentID> {
        if self.rewards_red.len() > self.rewards_blue.len(){
            Some(Blue)
        } else {
            if self.rewards_red.len() < self.max_rounds as usize{
                Some(Red)
            } else {
                None
            }
        }
    }

    fn is_finished(&self) -> bool {
        self.rewards_red.len()  >= self.max_rounds as usize
        && self.rewards_blue.len() >= self.max_rounds as usize
    }

    fn forward(&mut self, agent: DemoAgentID, action: DemoAction) -> Result<Self::Updates, DemoError> {
        if action.0 as usize > self.ceilings.len(){
            return Err(DemoError{})
        }
        let mut r = thread_rng();
        let d = Uniform::new(0.0, self.ceilings[action.0 as usize]);
        let reward: f32 = d.sample(&mut r);
        match agent{
            Blue => {
                self.rewards_blue.push(reward);
            }
            Red => {
                self.rewards_red.push(reward);
            }
        }


        Ok(vec![(agent, (agent, action.clone(), reward))])

    }
}



#[derive(Clone, Debug)]
pub struct DemoInfoSet{
    player_id: DemoAgentID,
    pub number_of_bandits: usize,
    rewards: Vec<f32>
}




impl DemoInfoSet{
    pub fn new(player_id: DemoAgentID, number_of_bandits: usize) -> Self{
        Self{
            player_id,
            number_of_bandits,
            rewards: Vec::new()
        }
    }
}

impl Renew<()> for DemoInfoSet{
    fn renew_from(&mut self, _base: ()) {
        self.rewards.clear()
    }
}

impl InformationSet<DemoDomain> for DemoInfoSet{
    fn agent_id(&self) -> &DemoAgentID {
        &self.player_id
    }


    fn is_action_valid(&self, action: &DemoAction) -> bool {
        (action.0 as usize) < self.number_of_bandits
    }

    fn update(&mut self, update: (DemoAgentID, DemoAction, f32)) -> Result<(), DemoError> {
        self.rewards.push(update.2);
        Ok(())
    }
}

impl PresentPossibleActions<DemoDomain> for DemoInfoSet{
    type ActionIteratorType = Vec<DemoAction>;

    fn available_actions(&self) -> Self::ActionIteratorType {
        let mut v = Vec::with_capacity(self.number_of_bandits);
        for i in 0..self.number_of_bandits as u8{
            v.push(DemoAction(i));
        }
        v
    }
}

impl EvaluatedInformationSet<DemoDomain> for DemoInfoSet{
    type RewardType = f32;

    fn current_subjective_score(&self) -> Self::RewardType {
        self.rewards.iter().sum()
    }

    fn penalty_for_illegal(&self) -> Self::RewardType {
        -100.0
    }
}

impl EnvironmentStateUniScore<DemoDomain> for DemoState{
    fn state_score_of_player(&self, agent: &DemoAgentID) -> f32 {
        match agent{
            Blue => {
                self.rewards_blue.iter().sum()
            },
            Red => {
                self.rewards_red.iter().sum()
            },
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DemoPolicySelectFirst{

}

impl Policy<DemoDomain> for DemoPolicySelectFirst{
    type InfoSetType = DemoInfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Option<DemoAction> {
        state.available_actions().first().map(|a| a.clone())
    }
}