use amfiteatr_core::agent::{InformationSet, PresentPossibleActions, EvaluatedInformationSet};
use amfiteatr_core::scheme::Scheme;
use std::fmt::Display;
use std::fmt::Formatter;
use crate::agent::ReplInfoSet;
use crate::AsymmetricRewardTableInt;
use crate::domain::{AgentNum, ClassicAction, ClassicGameDomain, ClassicGameDomainNumbered, ClassicGameError, IntReward};
use crate::domain::ClassicGameError::EncounterNotReported;

/// Information set of player that does not collect information about previous actions performed
/// and observed from enemy
#[derive(Copy, Clone, Debug)]
pub struct MinimalInfoSet {
    id: AgentNum,
    reward_table: AsymmetricRewardTableInt,
    payoff: IntReward

}

impl Display for MinimalInfoSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:}", self.id)
    }
}

impl MinimalInfoSet {
    pub fn new(id: AgentNum, reward_table: AsymmetricRewardTableInt) -> Self{
        Self{
            id, reward_table, payoff: 0
        }
    }
}

impl ReplInfoSet<AgentNum> for MinimalInfoSet{
    fn create(agent_id: AgentNum, reward_table: AsymmetricRewardTableInt) -> Self {
        Self::new(agent_id, reward_table)
    }
}






impl InformationSet<ClassicGameDomain<AgentNum>> for MinimalInfoSet {
    fn agent_id(&self) -> &AgentNum {
        &self.id
    }

    fn is_action_valid(&self, _action: &<ClassicGameDomain<AgentNum> as Scheme>::ActionType) -> bool {
        true
    }

    fn update(&mut self, update: <ClassicGameDomainNumbered as Scheme>::UpdateType) -> Result<(), ClassicGameError<AgentNum>> {

        if let Some(this_encounter_report) = update.encounters.get(&self.id){
            let reward = self.reward_table
                .reward_for_side(this_encounter_report.side, this_encounter_report.left_action(), this_encounter_report.right_action());

            self.payoff += reward;
            Ok(())
        } else{
            Err(EncounterNotReported(self.id))
        }
            //.ok_or(Err(EncounterNotReported(self.id as u32)));




    }
}

impl EvaluatedInformationSet<ClassicGameDomainNumbered, IntReward> for MinimalInfoSet {

    fn current_assessment(&self) -> IntReward {
        self.payoff
    }

    fn penalty_for_illegal(&self) -> IntReward {
        -10
    }
}

impl PresentPossibleActions<ClassicGameDomainNumbered> for MinimalInfoSet {
    type ActionIteratorType = [ClassicAction;2];

    fn available_actions(&self) -> Self::ActionIteratorType {
        [ClassicAction::Down, ClassicAction::Up]
    }
}