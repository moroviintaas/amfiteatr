use std::fmt::{Display, Formatter};
use log::trace;
use serde::Serialize;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_core::agent::{InformationSet, PresentPossibleActions, EvaluatedInformationSet};
use amfiteatr_core::scheme::{Renew};
use amfiteatr_core::error::{AmfiteatrError, ConvertError};
use amfiteatr_rl::tensor_data::{ContextEncodeTensor, TensorEncoding};
use crate::agent::{ActionPairMapper, AgentAssessmentClassic, EventCounts, ReplInfoSet};
use crate::AsymmetricRewardTableInt;
use crate::scheme::{AgentNum, AsUsize, ClassicAction, ClassicScheme, ClassicGameError, ClassicGameUpdate, EncounterReport, UsizeAgentId};
use crate::scheme::ClassicAction::{Down, Up};
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

    pub fn calculate_past_event_probabilities(&self) -> EventCounts {
        let mut counts = EventCounts::new();

        let mut previous = (None, None);
        let mut past_previous = (None, None);


        for report in &self.previous_encounters{

            if let (Some(p1), Some(p2)) = previous{
                if p2 == Up{
                    // other upped
                    if report.own_action == Up{
                        // we punish
                        counts.count_i_punish_immediately += 1.0;
                        if let (_, Some(pp2)) = past_previous{
                            if pp2 == Up{
                                counts.count_i_punish_after2 += 1.0;
                            }
                        }
                    }

                }
                if p1 == Up && report.other_player_action == Up {
                    counts.count_im_punished_immediately += 1.0;
                    if let (Some(pp1), _) = past_previous{
                        if pp1 == Up{
                            counts.count_im_punished_after2 += 1.0;
                        }
                    }
                }

                if p1 == Down && report.other_player_action == Down{
                    counts.count_im_absoluted_immediately += 1.0;
                }
                if p2 == Down && report.own_action == Down{
                    counts.count_i_absolute_immediately += 1.0;
                }

            }

            match (report.own_action, report.other_player_action){
                (Up, Up) => {
                    counts.count_up_v_up += 1.0;
                    past_previous = previous;
                    previous = (Some(Up), Some(Up))
                },
                (Up, Down) =>{
                    counts.count_up_v_down += 1.0;
                    past_previous = previous;
                    previous = (Some(Up), Some(Down));
                },
                (Down, Up) => {
                    counts.count_down_v_up += 1.0;
                    past_previous = previous;
                    previous = (Some(Down), Some(Up));
                },
                (Down, Down) => {
                    counts.count_down_v_down += 1.0;
                    past_previous = previous;
                    previous = (Some(Down), Some(Down))
                }
            }
        }
        let factor = self.previous_encounters.len() as f64;

        if !self.previous_encounters.is_empty(){

            counts.count_up_v_up /= factor;
            counts.count_down_v_up /= factor;
            counts.count_up_v_down /= factor;
            counts.count_down_v_down /= factor;

        }

        let f1 = factor - 1.0;
        if f1 > 0.0{
            counts.count_i_absolute_immediately /= f1;
            counts.count_im_absoluted_immediately /= f1;
            counts.count_i_punish_immediately /=f1;
            counts.count_im_punished_immediately /=f1;
        }

        let f2 = factor -2.0;
        if f2 >=0.0{
            counts.count_i_punish_after2 /=f2;
            counts.count_im_punished_after2 /= f2;
        }

        counts
        /*
        if let Some(first) =self.previous_encounters.first(){
            match (first.own_action,first.own_action){
                (Up, Up) => {
                    counts.count_up_v_up += 1.0;
                    previous = (Up, Up)
                },
                (Up, Down) =>{
                    counts.count_up_v_down += 1.0;
                    previous = (Up, Down);
                },
                (Down, Up) => {
                    counts.count_down_v_up += 1.0;
                    previous = (Down, Up);
                },
                (Down, Down) => {
                    counts.count_down_v_down += 1.0;
                    previous = (Down, Down)
                }
            }
        }

         */

    }


}

impl<ID: UsizeAgentId> ReplInfoSet<ID> for LocalHistoryInfoSet<ID>{
    fn create(agent_id: ID, reward_table: AsymmetricRewardTableInt) -> Self {
        Self::new(agent_id, reward_table)
    }
}

impl<ID: UsizeAgentId> Display for LocalHistoryInfoSet<ID> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Local History InfoSet:: Agent: {}, Rounds: {} ", self.id, self.previous_encounters.len())?;
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
        writeln!(f, "Previous observations: (c-c: {}, c-d: {}, d-c: {}, d-d: {})", self.count_actions[Down][Down],
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

impl<ID: UsizeAgentId> InformationSet<ClassicScheme<ID>> for LocalHistoryInfoSet<ID> {
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



impl TensorEncoding for LocalHistoryConversionToTensor {
    fn desired_shape(&self) -> &[i64] {
        &self.shape[..]
    }
}

/// Alias for info set for agents identified by `u32`.
pub type LocalHistoryInfoSetNumbered = LocalHistoryInfoSet<AgentNum>;

impl<ID: UsizeAgentId> ContextEncodeTensor<LocalHistoryConversionToTensor> for LocalHistoryInfoSet<ID>{
    fn try_to_tensor(&self, way: &LocalHistoryConversionToTensor) -> Result<Tensor, ConvertError> {
        let max_number_of_actions = way.shape()[1];
        if self.previous_encounters.len() > max_number_of_actions as usize{
            return Err(ConvertError::InfoSetNotFit {
                info_set: format!("Own encounter history information set with history of length {}, expected shape to fit = {:?}", self.previous_encounters.len(), way.shape()),
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

impl<ID: UsizeAgentId> PresentPossibleActions<ClassicScheme<ID>> for LocalHistoryInfoSet<ID>{
    type ActionIteratorType = [ClassicAction;2];

    fn available_actions(&self) -> Self::ActionIteratorType {
        [ClassicAction::Down, ClassicAction::Up]
    }
}

impl<ID: UsizeAgentId> Renew<ClassicScheme<ID>, ()> for LocalHistoryInfoSet<ID>{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ClassicScheme<ID>>> {
        self.previous_encounters.clear();
        self.cache_table_payoff = 0;
        self.count_actions = ActionPairMapper::zero();
        Ok(())
    }
}


impl<ID: UsizeAgentId> EvaluatedInformationSet<ClassicScheme<ID>, AgentAssessmentClassic<i64>> for LocalHistoryInfoSet<ID>{

    fn current_assessment(&self) ->AgentAssessmentClassic<i64> {

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

    fn penalty_for_illegal(&self) -> AgentAssessmentClassic<i64> {
        AgentAssessmentClassic::with_only_table_payoff(-100)
    }
}




