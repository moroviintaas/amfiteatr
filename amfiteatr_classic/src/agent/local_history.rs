use std::fmt::{Display, Formatter};
use log::trace;
use serde::Serialize;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_core::agent::{InformationSet, PresentPossibleActions, EvaluatedInformationSet};
use amfiteatr_core::domain::{Renew};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_rl::error::TensorRepresentationError;
use amfiteatr_rl::tensor_data::{CtxTryIntoTensor, ConversionToTensor};
use crate::agent::{ActionPairMapper, AgentAssessmentClassic};
use crate::AsymmetricRewardTableInt;
use crate::domain::{AgentNum, AsUsize, ClassicAction, ClassicGameDomain, ClassicGameError, ClassicGameUpdate, EncounterReport, UsizeAgentId};
use crate::domain::ClassicAction::{Down, Up};
use crate::Side::Left;


/// Information set for agent collecting previous encounter [`reports`](EncounterReport)
#[derive(Clone, Debug, Serialize)]
pub struct LocalHistoryInfoSet<ID: UsizeAgentId>{
    id: ID,
    previous_encounters: Vec<EncounterReport<ID>>,
    reward_table: AsymmetricRewardTableInt,
    count_actions: ActionPairMapper<i64>,
    cache_table_payoff: i64,

}

impl<ID: UsizeAgentId> LocalHistoryInfoSet<ID>{

    pub fn new(id: ID, reward_table: AsymmetricRewardTableInt) -> Self{
        Self{id, reward_table, previous_encounters: Default::default(), count_actions: Default::default(),
        cache_table_payoff: 0}
    }

    pub fn reset(&mut self){
        self.previous_encounters.clear();
        self.count_actions = ActionPairMapper::zero();
        self.cache_table_payoff = 0;
    }

    pub fn previous_encounters(&self) -> &Vec<EncounterReport<ID>>{
        &self.previous_encounters
    }

    pub fn count_actions_self_calculate(&self, action: ClassicAction) -> usize{
        self.previous_encounters.iter().filter(|e|{
            e.own_action == action
        }).count()
    }
    pub fn count_actions_other(&self, action: ClassicAction) -> usize{
        self.previous_encounters.iter().filter(|e|{
            e.other_player_action == action
        }).count()
    }
    pub fn action_counter(&self) -> &ActionPairMapper<i64>{
        &self.count_actions
    }
}

impl<ID: UsizeAgentId> Display for LocalHistoryInfoSet<ID> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Local History InfoSet:: Agent: {}, Rounds: {} \n", self.id, self.previous_encounters.len())?;
        /*let mut s = self.previous_encounters.iter().fold(String::new(), |mut acc, update| {
            acc.push_str(&format!("({:?}){:#}-{:#} #  ", update.side, update.own_action, update.other_player_action));
            acc
        });

         */
        for r in 0..self.previous_encounters.len(){
            let enc = &self.previous_encounters[r];
            write!(f, "\tround: {:3.}, paired against {},\tplayed {}\tagainst {};\t",
                r, ID::make_from_usize( enc.other_id.as_usize()),
                   enc.own_action, enc.other_player_action,
                    )?;
        }
        write!(f, "Current table payoff: {}.\t", self.cache_table_payoff,)?;
        write!(f, "Previous observations: (c-c: {}, c-d: {}, d-c: {}, d-d: {})\n", self.count_actions[Down][Down],
               self.count_actions[Down][Up],
               self.count_actions[Up][Down],
               self.count_actions[Up][Up])?;
        write!(f, "")
    }
}
/*
impl InformationSet<ClassicGameDomainNumbered> for OwnHistoryInfoSetNumbered{
    fn agent_id(&self) -> &AgentNum {
        &self.id
    }

    fn is_action_valid(&self, action: &ClassicAction) -> bool {
        true
    }

    fn update(&mut self, update: ClassicGameUpdate<AgentNum>) -> Result<(), ClassicGameError<AgentNum>> {
        let encounter = update.encounters[self.id as usize];
        self.previous_encounters.push(encounter);
        Ok(())
    }
}

 */

impl<ID: UsizeAgentId> InformationSet<ClassicGameDomain<ID>> for LocalHistoryInfoSet<ID> {
    fn agent_id(&self) -> &ID {
        &self.id
    }

    fn is_action_valid(&self, _action: &ClassicAction) -> bool {
        true
    }

    fn update(&mut self, update: ClassicGameUpdate<ID>) -> Result<(), ClassicGameError<ID>> {

        let report = update.encounters[&self.id];
        match report.own_action  {
            Down => match report.other_player_action{
                Up => self.count_actions[Down][Up] += 1,
                Down => self.count_actions[Down][Down] += 1,
            },
            Up => match report.other_player_action{
                Up => self.count_actions[Up][Up] += 1,
                Down => self.count_actions[Up][Down] += 1,
            },
        }
        self.previous_encounters.push(report);
        self.cache_table_payoff += report.calculate_reward(&self.reward_table);
        trace!("After info set update on agent {}, with {} previous actions", self.agent_id(), self.previous_encounters.len());
        Ok(())
    }
}
/*
impl<ID: UsizeAgentId> ScoringInformationSet<ClassicGameDomain<ID>> for OwnHistoryInfoSet<ID>{
    type RewardType = i32;

    fn current_subjective_score(&self) -> Self::RewardType {
        self.previous_encounters.iter().map(|r|{
            r.calculate_reward(&self.reward_table)
        }).sum()
    }

    fn penalty_for_illegal(&self) -> Self::RewardType {
        -100
    }
}

 */
/// Represents way how information set should be represented as tensor including information about
/// previous action made by player and enemy.
#[derive(Copy, Clone, Debug, Default)]
pub struct LocalHistoryConversionToTensor {
    shape: [i64; 2]
}

impl LocalHistoryConversionToTensor {
    pub fn new(number_of_rounds: usize) -> Self{
        Self{
            shape: [2, number_of_rounds as i64]
        }
    }
    /// Returns expected shape of tensor which is `[2, number_of_rounds]`.
    /// Two rows to store own action in first and enemy action in second
    pub fn shape(&self) -> &[i64]{
        &self.shape[..]
    }
}



impl ConversionToTensor for LocalHistoryConversionToTensor {
    fn desired_shape(&self) -> &[i64] {
        &self.shape[..]
    }
}

/// Alias for info set for agents identified by `u32`.
pub type LocalHistoryInfoSetNumbered = LocalHistoryInfoSet<AgentNum>;

impl<ID: UsizeAgentId> CtxTryIntoTensor<LocalHistoryConversionToTensor> for LocalHistoryInfoSet<ID>{
    fn try_to_tensor(&self, way: &LocalHistoryConversionToTensor) -> Result<Tensor, TensorRepresentationError> {
        let max_number_of_actions = way.shape()[1];
        if self.previous_encounters.len() > max_number_of_actions as usize{
            return Err(TensorRepresentationError::InfoSetNotFit {
                info_set: format!("Own encounter history information set with history of length {}", self.previous_encounters.len()),
                shape: Vec::from(way.shape()),
            });
        }
        let mut own_actions: Vec<f32> = self.previous_encounters.iter().map(|e|{
            e.own_action.as_usize() as f32
        }).collect();
        own_actions.resize_with(max_number_of_actions as usize, ||-1.0);
        let mut other_actions: Vec<f32> = self.previous_encounters.iter().map(|e|{
            e.other_player_action.as_usize()  as f32
        }).collect();
        other_actions.resize_with(max_number_of_actions as usize, ||-1.0);

        let own_tensor = Tensor::f_from_slice(&own_actions[..])?;
        let other_tensor = Tensor::f_from_slice(&other_actions[..])?;

        let result = Tensor::f_stack(&[own_tensor, other_tensor], 0)?
            .flatten(0, -1);
        Ok(result)

    }
}

impl<ID: UsizeAgentId> PresentPossibleActions<ClassicGameDomain<ID>> for LocalHistoryInfoSet<ID>{
    type ActionIteratorType = [ClassicAction;2];

    fn available_actions(&self) -> Self::ActionIteratorType {
        [ClassicAction::Down, ClassicAction::Up]
    }
}

impl<ID: UsizeAgentId> Renew<ClassicGameDomain<ID>, ()> for LocalHistoryInfoSet<ID>{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ClassicGameDomain<ID>>> {
        self.previous_encounters.clear();
        self.cache_table_payoff = 0;
        self.count_actions = ActionPairMapper::zero();
        Ok(())
    }
}


impl<ID: UsizeAgentId> EvaluatedInformationSet<ClassicGameDomain<ID>,> for LocalHistoryInfoSet<ID>{
    type RewardType = AgentAssessmentClassic<i64>;

    fn current_subjective_score(&self) -> Self::RewardType {

        let mut edu_asses = 0.0;
        if self.previous_encounters.len() >=2{
            for i in 0..(self.previous_encounters.len()-1){

                if self.previous_encounters[i].other_player_action == Up
                    && self.previous_encounters[i+1].own_action == Up {
                    edu_asses += 0.1;
                }
                if self.previous_encounters[i].other_player_action == Down
                    && self.previous_encounters[i+1].own_action == Down {
                    edu_asses += 1.0 +
                        (self.reward_table.reward_for_side(Left, Up, Down)
                            - self.reward_table.reward_for_side(Left, Down, Down)) as f32;
                }
            }
        }

        if let Some(prev) = self.previous_encounters.last(){
            if prev.own_action == Down && prev.other_player_action == Down {
                 edu_asses += 1.0 +
                        (self.reward_table.reward_for_side(Left, Up, Down)
                            - self.reward_table.reward_for_side(Left, Down, Down)) as f32;
            }
        }


        AgentAssessmentClassic::new(self.cache_table_payoff, self.count_actions, edu_asses)
    }

    fn penalty_for_illegal(&self) -> Self::RewardType {
        AgentAssessmentClassic::with_only_table_payoff(-100)
    }
}


