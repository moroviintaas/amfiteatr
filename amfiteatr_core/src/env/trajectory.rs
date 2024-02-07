use std::fmt::{Display, Formatter};
use std::ops::Index;
pub use crate::agent::Trajectory;
use crate::env::EnvironmentStateSequential;
use crate::domain::DomainParameters;


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