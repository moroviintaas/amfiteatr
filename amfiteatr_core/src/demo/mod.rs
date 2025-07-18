//! # Minimal example
//! ```
//! use std::collections::{HashMap, HashSet};
//! use std::thread;
//! use amfiteatr_core::agent::{AgentGen, TracingAgentGen,  AutomaticAgent, RewardedAgent, RandomPolicy};
//! use amfiteatr_core::comm::StdEnvironmentEndpoint;
//! use amfiteatr_core::demo::{DemoInfoSet, DemoDomain, DemoState, DemoPolicySelectFirst, DEMO_AGENT_BLUE, DEMO_AGENT_RED};
//! use amfiteatr_core::env::*;
//!
//!
//! let bandits = vec![(0.0, 5.0), (0.0, 11.5), (0.0, 6.0)];
//! let number_of_bandits = bandits.len();
//!
//! let (comm_env_blue, comm_agent_blue) = StdEnvironmentEndpoint::new_pair();
//! let (comm_env_red, comm_agent_red) = StdEnvironmentEndpoint::new_pair();
//! let mut env_comms = HashMap::new();
//! env_comms.insert(DEMO_AGENT_BLUE, comm_env_blue);
//! env_comms.insert(DEMO_AGENT_RED, comm_env_red);
//! let player_set = env_comms.keys().map(|id| *id).collect();
//! let state = DemoState::new_with_players(bandits, 100, &player_set);
//! let mut environment = TracingHashMapEnvironment::new(state, env_comms);
//! let blue_info_set = DemoInfoSet::new(DEMO_AGENT_BLUE, number_of_bandits);
//! let red_info_set = DemoInfoSet::new(DEMO_AGENT_RED, number_of_bandits);
//! let mut agent_blue = TracingAgentGen::new(blue_info_set, comm_agent_blue, RandomPolicy::<DemoDomain, DemoInfoSet>::new());
//! let mut agent_red = AgentGen::new(red_info_set, comm_agent_red, DemoPolicySelectFirst{});
//!
//! thread::scope(|s|{
//!     s.spawn(||{
//!         environment.run_round_robin_with_rewards().unwrap();
//!     });
//!     s.spawn(||{
//!         agent_blue.run().unwrap();
//!     });
//!     s.spawn(||{
//!         agent_red.run().unwrap();
//!     });
//! });
//!
//! assert_eq!(environment.trajectory().number_of_steps(), 200);
//! assert!(environment.actual_score_of_player(&DEMO_AGENT_BLUE) > 10.0);
//! assert!(agent_blue.current_universal_score() > 10.0);
//! assert!(agent_red.current_universal_score() > 10.0);
//! assert!(agent_blue.current_universal_score() > agent_red.current_universal_score());
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use nom::IResult;
use rand::distr::Uniform;
use crate::agent::{AgentIdentifier, Policy, PresentPossibleActions};
use crate::domain::{Action, DomainParameters, Renew};
use crate::env::{SequentialGameState, GameStateWithPayoffs};
use rand::distr::Distribution;
use rand::{rng};
use crate::agent::{InformationSet, EvaluatedInformationSet};
use crate::error::AmfiteatrError;
use crate::util::{StrParsed};

pub const DEMO_AGENT_BLUE: DemoAgentID = 0;
pub const DEMO_AGENT_RED: DemoAgentID = 1;



#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub struct DemoAction(pub u8);
impl Display for DemoAction{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl StrParsed for DemoAction{
    fn parse_from_str(input: & str) -> IResult<&str, Self> {
        nom::character::complete::u8(input).map(|(rest, n)|(rest, Self(n)))
    }
}

impl Action for DemoAction{}

/*
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, )]
pub enum DemoAgentID{
    Blue,
    Red
}*/
pub type DemoAgentID = usize;
/*
impl Display for DemoAgentID{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

 */



impl AgentIdentifier for DemoAgentID{}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub struct DemoError(pub String);
impl Display for DemoError{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DemoError: {}", self.0)
    }
}



#[derive(Clone, Debug, PartialEq, Eq)]
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
    bounds: Vec<(f32, f32)>,
    max_rounds: usize,
    //rewards_red: Vec<f32>,
    //rewards_blue: Vec<f32>,
    rewards: HashMap<DemoAgentID, Vec<f32>>,
    player_ids: Vec<DemoAgentID>,
    turn_of: Option<usize>
}

impl DemoState{

    /*
    pub fn new(ceilings: Vec<f32>, max_rounds: u32) -> Self{
        Self{ceilings, max_rounds, rewards: HashMap::new(), player_indexes: Vec::new(), turn_of: None }
    }
    */
    pub fn new_with_players(bounds: Vec<(f32, f32)>, max_rounds: usize, comms: &HashSet<DemoAgentID>) -> Self{
        let player_ids: Vec<DemoAgentID> = comms.iter().copied().collect();
        let turn_of  = if max_rounds > 0{
            Some(0)
        } else {
            None
        };
        let rewards = player_ids.iter().map(|id| (id.to_owned(), Vec::new())).collect();
        Self{
            bounds, max_rounds, rewards, player_ids, turn_of
        }

    }
}
impl SequentialGameState<DemoDomain> for DemoState{
    type Updates = Vec<(DemoAgentID, (DemoAgentID, DemoAction, f32))>;

    fn current_player(&self) -> Option<DemoAgentID> {
        /*
        if self.rewards_red.len() > self.rewards_blue.len(){
            Some(Blue)
        } else {
            if self.rewards_red.len() < self.max_rounds as usize{
                Some(Red)
            } else {
                None
            }
        }*/
        self.turn_of.map(|index| self.player_ids[index])
    }

    fn is_finished(&self) -> bool {
        /*
        self.rewards_red.len()  >= self.max_rounds as usize
        && self.rewards_blue.len() >= self.max_rounds as usize
        */
        self.turn_of.is_none()

    }

    fn forward(&mut self, agent: DemoAgentID, action: DemoAction) -> Result<Self::Updates, DemoError> {

        if action.0 as usize > self.bounds.len(){
            return Err(DemoError(format!("Agent used {}'th bandit which is not defined", action.0)));
        }
        if let Some(current_player_index) = self.turn_of{
            if self.player_ids[current_player_index] != agent{
                return Err(DemoError(format!("Bad player order, expected: {}, received: {agent}", &self.player_ids[current_player_index] )));
            }

            let mut r = rng();
            let reward = if self.bounds[action.0 as usize].0 == self.bounds[action.0 as usize].1{
                self.bounds[action.0 as usize].0
            } else {
                let d = Uniform::new(self.bounds[action.0 as usize].0, self.bounds[action.0 as usize].1)
                    .map_err(|e| DemoError(e.to_string()))?;
                d.sample(&mut r)
            };

            self.rewards.get_mut(&agent).unwrap().push(reward);



            let mut next_player_index = current_player_index+1;
            if next_player_index >= self.player_ids.len(){
                next_player_index = 0;
            }
            if self.rewards[&self.player_ids[next_player_index]].len() >= self.max_rounds{
                self.turn_of = None
            } else {
                self.turn_of = Some(next_player_index)
            }

            let updates = self.player_ids.iter().map(|id|{
                (*id, (agent, action.clone(), reward))
            }).collect();

            Ok(updates)


        } else {
            Err(DemoError(format!("Player {} played, while game is finished", agent)))
        }




    }
}

impl Renew<DemoDomain, ()> for DemoState{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<DemoDomain>> {
        self.turn_of = if self.max_rounds > 0{
            Some(0)
        } else {
            None
        };
        for (_, r) in self.rewards.iter_mut(){
            r.clear()
        }

        Ok(())
    }
}



#[derive(Clone, Debug, PartialEq)]
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

impl Renew<DemoDomain, ()> for DemoInfoSet{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<DemoDomain>> {
        self.rewards.clear();
        Ok(())
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

        if self.player_id == update.0{
            self.rewards.push(update.2);

        } else {
            #[cfg(feature = "log_trace")]
            log::trace!("Update of other player's action")
        }
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

impl EvaluatedInformationSet<DemoDomain, f32> for DemoInfoSet{

    fn current_assessment(&self) -> f32 {
        self.rewards.iter().sum()
    }

    fn penalty_for_illegal(&self) -> f32 {
        -100.0
    }
}

impl GameStateWithPayoffs<DemoDomain> for DemoState{
    fn state_payoff_of_player(&self, agent: &DemoAgentID) -> f32 {
        /*
        match agent{
            Blue => {
                self.rewards_blue.iter().sum()
            },
            Red => {
                self.rewards_red.iter().sum()
            },
        }

         */

        self.rewards[agent].iter().sum()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DemoPolicySelectFirst{

}

impl Policy<DemoDomain> for DemoPolicySelectFirst{
    type InfoSetType = DemoInfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DemoAction, AmfiteatrError<DemoDomain>> {
        state.available_actions().first().cloned().ok_or_else(|| AmfiteatrError::NoActionAvailable {
            context: "Demo Policy".into()
        })
    }
}


/*
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub struct DemoComplexAction(pub u8);

 */