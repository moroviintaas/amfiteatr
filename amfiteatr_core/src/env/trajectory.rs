use std::fmt::{Display, Formatter};
use std::ops::Index;
use crate::env::EnvironmentStateSequential;
use crate::domain::DomainParameters;
use crate::error::{AmfiteatrError, TrajectoryError};

/*
/// Trace step of while game (traced by game environment)
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct EnvironmentTraceStep<DP: DomainParameters, S: EnvironmentStateSequential<DP>>{
    state_before: S,
    agent: DP::AgentId,
    action: DP::ActionType,
    is_action_valid: bool
}

impl<DP: DomainParameters, S: EnvironmentStateSequential<DP>> Display for EnvironmentTraceStep<DP, S>
where S: Display, <DP as DomainParameters>::AgentId: Display,
      <DP as DomainParameters>::ActionType: Display{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ {} ][ {} / {} ", self.state_before, self.agent, self.action)?;
        if !self.is_action_valid{
            write!(f, "!]")
        } else {
            write!(f, "]")
        }
    }
}


impl<DP: DomainParameters, S: EnvironmentStateSequential<DP>> EnvironmentTraceStep<DP, S>{

    pub fn new(state_before: S, agent: DP::AgentId,
               action: DP::ActionType, is_valid: bool) -> Self{
        /*let checked_action = match is_valid{
            false => CheckedAction::Invalid(action),
            true => CheckedAction::Valid(action)
        };

         */
        Self{state_before, agent, action, is_action_valid: is_valid}
    }

    pub fn state_before(&self) -> &S{
        &self.state_before
    }

    pub fn agent(&self) -> &DP::AgentId{
        &self.agent
    }

    pub fn action(&self) -> &DP::ActionType{
        &self.action
    }


    pub fn is_action_valid(&self) -> bool{
        self.is_action_valid
    }
}

/// Standard trajectory for environment
//pub type StdEnvironmentTrajectory<DP, S> = Trajectory<EnvironmentTraceStep<DP, S>>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
pub struct EnvironmentTrajectory<DP: DomainParameters, S: EnvironmentStateSequential<DP>> {


    #[cfg_attr(feature = "serde",
        serde(bound(serialize =
            "DP::ActionType: serde::Serialize, \
            DP::UniversalReward:serde::Serialize, \
            DP::AgentId: serde::Serialize"
        )))
    ]
    #[cfg_attr(feature = "serde",
        serde(bound(deserialize =
            "DP::ActionType: serde::Deserialize<'de>, \
            DP::UniversalReward: serde::Deserialize<'de>, \
            DP::AgentId: serde::Deserialize<'de>"
        )))
    ]
    history: Vec<EnvironmentTraceStep<DP, S>>,
    //revoked_steps: Vec<AgentTraceStep<DP, S>>,
    final_state: Option<S>,

}

impl<DP: DomainParameters, S: EnvironmentStateSequential<DP>> Default for EnvironmentTrajectory<DP, S>{
    fn default() -> Self {
        Self{
            history: Vec::new(),
            final_state: None
        }
    }
}
impl<DP: DomainParameters, S: EnvironmentStateSequential<DP>>  EnvironmentTrajectory<DP, S>
{


    pub fn new() -> Self{
        Self{ history: Default::default(), final_state: Default::default()}
    }
    /*pub fn register_line(&mut self, state: S, action: DP::ActionType, reward_for_action: S::RewardType){
        self.trace.push(GameTraceLine::new(state, action, reward_for_action));

    }*/
    pub fn new_reserve(capacity: usize) -> Self{
        Self{ history: Vec::with_capacity(capacity), final_state: Default::default()}
    }

    /// Pushes trace step on the end of trajectory.
    pub fn push_trace_step(&mut self, trace_step: EnvironmentTraceStep<DP, S>){
        self.history.push(trace_step);
    }
    /// Clears trajectory using [`Vec::clear()`](std::vec::Vec::clear)
    pub fn clear(&mut self){
        self.history.clear();
        self.final_state = None;
    }

    /// Returns reference to `Vec` inside the structure.
    pub fn list(&self) -> &Vec<EnvironmentTraceStep<DP, S>>{
        &self.history
    }

    /// Pops step from trajectory using [`Vec::pop()`](std::vec::Vec::pop)
    pub fn pop_step(&mut self) -> Option<EnvironmentTraceStep<DP, S>>{
        self.history.pop()
    }


    pub fn is_empty(&self) -> bool{
        self.list().is_empty()
    }

    pub fn finalize(&mut self, state: S){
        self.final_state = Some(state);
    }

    pub fn final_state(&self) -> &Option<S>{
        &self.final_state
    }

}

impl<DP: DomainParameters, S: EnvironmentStateSequential<DP>> Index<usize> for EnvironmentTrajectory<DP, S>{
    type Output = EnvironmentTraceStep<DP, S>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.history[index]
    }
}

 */

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct GameStepView<'a, DP: DomainParameters, S: EnvironmentStateSequential<DP>>{
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a S: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a S: serde::Serialize")))]
    state_before: &'a S,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a DP::AgentId: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a DP::AgentId: serde::Serialize")))]
    agent: &'a DP::AgentId,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a DP::ActionType: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a DP::ActionType: serde::Serialize")))]
    action: &'a DP::ActionType,
    is_action_valid: bool,

    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a S: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a S: serde::Serialize")))]
    state_after: &'a S,



}

impl<'a, DP: DomainParameters, S: EnvironmentStateSequential<DP>> Display for GameStepView<'a, DP, S>
    where &'a S: Display, <DP as DomainParameters>::AgentId: Display,
           &'a <DP as DomainParameters>::ActionType: Display{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ {} ]--[ {} / {}]-->[ {} ] ", self.state_before, self.agent, self.action, self.state_after)?;
        if !self.is_action_valid{
            write!(f, "!]")
        } else {
            write!(f, "]")
        }
    }
}

impl<'a, DP: DomainParameters, S: EnvironmentStateSequential<DP>> GameStepView<'a, DP, S>{

    pub fn new(state_before: &'a S, agent: &'a DP::AgentId,
               action: &'a DP::ActionType, is_valid: bool, state_after: &'a S) -> Self{
        Self{state_before, agent, action, is_action_valid: is_valid, state_after}
    }

    pub fn state_before(&self) -> &S{
        &self.state_before
    }

    pub fn agent(&self) -> &DP::AgentId{
        &self.agent
    }

    pub fn action(&self) -> &DP::ActionType{
        &self.action
    }


    pub fn is_action_valid(&self) -> bool{
        self.is_action_valid
    }
}
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct GameTrajectory<DP: DomainParameters, S: EnvironmentStateSequential<DP>>{
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "S: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "S: serde::Serialize")))]
    states: Vec<S>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::AgentId: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::AgentId: serde::Serialize")))]
    agents: Vec<DP::AgentId>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::ActionType: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::ActionType: serde::Serialize")))]
    actions: Vec<DP::ActionType>,
    validations: Vec<bool>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "S: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "S: serde::Serialize")))]
    final_state: Option<S>,
}
impl<DP: DomainParameters, S: EnvironmentStateSequential<DP>> Default for GameTrajectory<DP, S>{
    fn default() -> Self {
        Self{
            states: vec![],
            agents: vec![],
            actions: vec![],
            validations: vec![],
            final_state: None,
        }
    }
}
impl<DP: DomainParameters, S: EnvironmentStateSequential<DP>> GameTrajectory<DP, S>{

    pub fn new() -> Self{
        Default::default()
    }

    pub fn clear(&mut self){
        self.actions.clear();
        self.states.clear();
        self.actions.clear();
        self.final_state = None;
    }

    pub fn completed_len(&self) -> usize{
        match self.is_finished(){
            true => self.states.len(),
            false => self.states.len().saturating_sub(1)
        }
    }
    pub fn register_step(&mut self, state: S, agent: DP::AgentId, action: DP::ActionType, is_valid: bool)
                         -> Result<(), AmfiteatrError<DP>>{
        if self.final_state.is_some(){
            return Err(TrajectoryError::UpdateOnFinishedGameTrajectory {
                description: format!("{:?}", self)
            }.into());
        }
        self.states.push(state);
        self.actions.push(action);
        self.agents.push(agent);
        self.validations.push(is_valid);

        Ok(())
    }
    pub fn finish(&mut self, state: S) -> Result<(), AmfiteatrError<DP>>{
        if self.final_state.is_some(){
            return Err(TrajectoryError::FinishingOnFinishedGameTrajectory {
                description: format!("{:?}", &self)}.into())
        }
        self.final_state = Some(state);
        Ok(())
    }
    pub fn is_finished(&self) -> bool{
        self.final_state.is_some()
    }
    pub fn is_empty(&self) -> bool {
        self.states.is_empty() || self.validations.is_empty() || self.agents.is_empty() || self.actions.is_empty()
    }

    pub fn last_view_step(&self) -> Option<GameStepView<DP, S>>{
        if self.final_state.is_some(){
            self.view_step(self.states.len().saturating_sub(1))
        } else {
            self.view_step(self.states.len().saturating_sub(2))
        }
    }
    pub fn view_step(&self, index: usize) -> Option<GameStepView<DP, S>>{
        self.states.get(index+1).and_then(|se|{
            Some(GameStepView::new(
                &self.states[index],
                &self.agents[index],
                &self.actions[index],
                self.validations[index],
                &se))
        }).or_else(||{
            if index + 1 == self.states.len(){
                if let Some(fi_state) = &self.final_state{
                    Some(GameStepView::new(
                        &self.states[index],
                        &self.agents[index],
                        &self.actions[index],
                        self.validations[index],
                        fi_state
                    ))
                } else {
                    None
                }
            } else { None}
        })
    }
}