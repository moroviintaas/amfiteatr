use std::fmt::{Debug, Display, Formatter};
use std::ops::Index;
use crate::agent::info_set::EvaluatedInformationSet;
use crate::domain::DomainParameters;


/// This struct contains information about _information set (game state from view of agent)_
/// before taken action along with taken action and saved score before and after taking action.
/// __Note__ scores after taking action are __not__ measured in the moment just after taking action,
/// but just before taking subsequent action i.e. this is _information set_ for __next__ step.
///

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct AgentTraceStep<DP: DomainParameters, S: EvaluatedInformationSet<DP>> {
    initial_info_set: S,
    taken_action: DP::ActionType,
    initial_payoff: DP::UniversalReward,
    updated_payoff: DP::UniversalReward,

    initial_subjective_assessment: S::RewardType,
    updated_subjective_assessment: S::RewardType


}



impl<DP: DomainParameters, S: EvaluatedInformationSet<DP>> AgentTraceStep<DP, S>
//where for <'a> &'a<DP as DomainParameters>::UniversalReward: Sub<&'a <DP as DomainParameters>::UniversalReward, Output=<DP as DomainParameters>::UniversalReward>,
//    for<'a> &'a <S as ScoringInformationSet<DP>>::RewardType: Sub<&'a  <S as ScoringInformationSet<DP>>::RewardType, Output = <S as ScoringInformationSet<DP>>::RewardType>

{
    /// Constructor of AgentTraceStep
    /// # Args:
    /// - `initial_info_set`: Information set before taken action
    /// - `taken_action`: Performed action (in the state of `initial_info_set`)
    /// - `initial_universal_state_score`: score before taking action, i.e. at the moment of `initial_info_set`,
    ///  taken from environment
    /// - `updated_universal_state_score`: score after taking action
    /// taken from environment
    /// - `initial_subjective_state_score`: score before taking action, i.e. at the moment of `initial_info_set`
    /// measured on information set
    /// - `updated_universal_state_score`: score after taking action - measured on information set
    pub fn new(

        initial_info_set: S,
        taken_action: DP::ActionType,
        initial_universal_state_score: DP::UniversalReward,
        updated_universal_state_score: DP::UniversalReward,
        initial_subjective_state_score: S::RewardType,
        updated_subjective_state_score: S::RewardType
    ) -> Self{
        Self {
            initial_info_set,
            taken_action,
            initial_payoff: initial_universal_state_score,
            updated_payoff: updated_universal_state_score,
            initial_subjective_assessment: initial_subjective_state_score,
            updated_subjective_assessment: updated_subjective_state_score
        }
    }

    /// Returns reference to information set trapped for this step (before action taken)
    pub fn step_info_set(&self) -> &S{
        &self.initial_info_set
    }

    /// Return reference to taken action in this step
    pub fn taken_action(&self) -> &DP::ActionType{
        &self.taken_action
    }

    /// Returns subjective reward for taken action - difference between score before __next__ action,
    /// and score before taking __this__ action. This relates to reward received from environment.
    pub fn step_subjective_reward(&self) -> S::RewardType{
        let n = self.updated_subjective_assessment.clone();
        n - &self.initial_subjective_assessment

    }
    /// Returns subjective reward for taken action - difference between score before __next__ action,
    /// and score before taking __this__ action. This relates to reward measured on information set.
    pub fn step_universal_reward(&self) -> DP::UniversalReward{
        let n = self.updated_payoff.clone();
        n - &self.initial_payoff
    }

    /// Returns reference universal score (sourced from environment)
    pub fn universal_score_before(&self) -> &DP::UniversalReward{
        &self.initial_payoff
    }
    /// Returns reference to score sourced from information set (before action)
    pub fn subjective_score_before(&self) -> &S::RewardType{
        &self.initial_subjective_assessment
    }


    /// Returns reference to universal score (sourced from environment) after taking action (and optional actions of other players
    pub fn universal_score_after(&self) -> &DP::UniversalReward{
        &self.updated_payoff
    }

    /// Returns reference to subjective score (sourced from information set) after taking action (and optional actions of other players
    pub fn subjective_score_after(&self) -> &S::RewardType{
        &self.updated_subjective_assessment
    }



    /// Returns tuple of respectively: reference to information set, reference to taken action, reward for taken action (sourced from environment)
    pub fn s_a_r_universal(&self) -> (&S, &DP::ActionType, DP::UniversalReward) {
        (self.step_info_set(), self.taken_action(), self.step_universal_reward())
    }

    /// Returns tuple of respectively: reference to information set, reference to taken action, reward for taken action (sourced from information set)
    pub fn s_a_r_subjective(&self) -> (&S, &DP::ActionType, S::RewardType) {
        (self.step_info_set(), self.taken_action(), self.step_subjective_reward())
    }
    //pub fn s_a_r(&self, source:S RewardSource) -
}

impl<DP: DomainParameters, S: EvaluatedInformationSet<DP>> Display for AgentTraceStep<DP, S>
where
    S: Display,
    <DP as DomainParameters>::UniversalReward: Display,
    <DP as DomainParameters>::ActionType: Display,
    <S as  EvaluatedInformationSet<DP>>::RewardType : Display{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[State: {} ][From Score: U = {} | A = {}][Action: {} ][To Score: U = {} | A = {}]",
               self.initial_info_set,

               self.initial_payoff,
               self.initial_subjective_assessment,
               self.taken_action,
               self.updated_payoff,
               self.updated_subjective_assessment
        )
    }
}


/// Trajectory of game from the view of agent.
/// > This structure needs a rework in future. For now it wraps around `Vec` of steps, where single
/// step typically consists of initial state, taken action, and rewards. It may be helpful to make
/// single steps have access to another steps (to allow some difference analysis) - in such case
/// we would like some counted reference ([`Arc`](std::sync::Arc)).
/// Second problem is that current implementation does not capture final state in trajectory.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
pub struct Trajectory<DP: DomainParameters, S: EvaluatedInformationSet<DP>> {


    //top_state: S,
    #[cfg_attr(feature = "serde",
        serde(bound(serialize =
            "DP::ActionType: serde::Serialize, \
            DP::UniversalReward: serde::Serialize, \
            <S as EvaluatedInformationSet<DP>>::RewardType: serde::Serialize"))
        )
    ]
    #[cfg_attr(feature = "serde",
    serde(bound(deserialize =
    "DP::ActionType: serde::Deserialize<'de>, \
            DP::UniversalReward: serde::Deserialize<'de>, \
            <S as EvaluatedInformationSet<DP>>::RewardType: serde::Deserialize<'de>"))
        )
    ]
    history: Vec<AgentTraceStep<DP, S>>,
    //revoked_steps: Vec<AgentTraceStep<DP, S>>,
    final_information_set: Option<S>,

}
/// This is proposed default agent trajectory, where single step is of type
/// [`AgentTraceStep`]
//pub type StdAgentTrajectory<DP, IS> = Trajectory<AgentTraceStep<DP, IS>>;
impl<DP: DomainParameters, S: EvaluatedInformationSet<DP>>  Default for Trajectory<DP, S>{
    fn default() -> Self {
        Self{ history: Default::default(), final_information_set: Default::default()}
    }
}
impl<DP: DomainParameters, S: EvaluatedInformationSet<DP>>  Trajectory<DP, S>
{


    pub fn new() -> Self{
        Self{ history: Default::default(), final_information_set: Default::default()}
    }
    /*pub fn register_line(&mut self, state: S, action: DP::ActionType, reward_for_action: S::RewardType){
        self.trace.push(GameTraceLine::new(state, action, reward_for_action));

    }*/
    pub fn new_reserve(capacity: usize) -> Self{
        Self{ history: Vec::with_capacity(capacity), final_information_set: Default::default()}
    }

    /// Pushes trace step on the end of trajectory.
    pub fn push_trace_step(&mut self, trace_step: AgentTraceStep<DP, S>){
        self.history.push(trace_step);
    }
    /// Clears trajectory using [`Vec::clear()`](std::vec::Vec::clear)
    pub fn clear(&mut self){
        self.history.clear();
        self.final_information_set = None;
    }

    /// Returns reference to `Vec` inside the structure.
    pub fn list(&self) -> &Vec<AgentTraceStep<DP, S>>{
        &self.history
    }

    /// Pops step from trajectory using [`Vec::pop()`](std::vec::Vec::pop)
    pub fn pop_step(&mut self) -> Option<AgentTraceStep<DP, S>>{
        self.history.pop()
    }


    pub fn is_empty(&self) -> bool{
        self.list().is_empty()
    }

    pub fn finalize(&mut self, information_set: S){
        self.final_information_set = Some(information_set);
    }
    pub fn final_information_set(&self) -> &Option<S>{
        &self.final_information_set
    }
}

impl<DP: DomainParameters, S: EvaluatedInformationSet<DP>> Index<usize> for Trajectory<DP, S>{
    type Output = AgentTraceStep<DP, S>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.history[index]
    }
}