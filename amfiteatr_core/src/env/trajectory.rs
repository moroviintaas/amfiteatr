use std::fmt::{Display, Formatter};
use crate::env::SequentialGameState;
use crate::scheme::Scheme;
use crate::error::{AmfiteatrError, TrajectoryError};



/// View of single step in game trajectory.
///
/// Provides references to states
/// both for time moment before step and at the moment before next step.
/// View contains also action done and agent performing it in this step.
/// Game step is measured from one state to the next (transaction is made by any agent performing action).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct GameStepView<'a, DP: Scheme, ST: SequentialGameState<DP>>{
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a ST: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a ST: serde::Serialize")))]
    state_before: &'a ST,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a DP::AgentId: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a DP::AgentId: serde::Serialize")))]
    agent: &'a DP::AgentId,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a DP::ActionType: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a DP::ActionType: serde::Serialize")))]
    action: &'a DP::ActionType,
    is_action_valid: bool,

    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a ST: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a ST: serde::Serialize")))]
    state_after: &'a ST,



}

impl<'a, DP: Scheme, ST: SequentialGameState<DP>> Display for GameStepView<'a, DP, ST>
    where &'a ST: Display, <DP as Scheme>::AgentId: Display,
           &'a <DP as Scheme>::ActionType: Display{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ {} ]--[ {} / {}]-->[ {} ] ", self.state_before, self.agent, self.action, self.state_after)?;
        if !self.is_action_valid{
            write!(f, "!]")
        } else {
            write!(f, "]")
        }
    }
}

impl<'a, DP: Scheme, ST: SequentialGameState<DP>> GameStepView<'a, DP, ST>{

    pub fn new(state_before: &'a ST, agent: &'a DP::AgentId,
               action: &'a DP::ActionType, is_valid: bool, state_after: &'a ST) -> Self{
        Self{state_before, agent, action, is_action_valid: is_valid, state_after}
    }

    /// Returns state before taking action.
    pub fn state(&self) -> &ST{
        self.state_before
    }
    /// Returns state after taking action.
    pub fn late_state(&self) -> &ST{
        self.state_after
    }

    /// Returns reference to the agent id who performed action in the step's early state.
    pub fn agent(&self) -> &DP::AgentId{
        self.agent
    }

    /// Returns reference to action performed during the step
    pub fn action(&self) -> &DP::ActionType{
        self.action
    }


    pub fn is_action_valid(&self) -> bool{
        self.is_action_valid
    }
}


/// Game trajectory, keeping information about following _states_, and _actions_ performed by _agents_.
///
/// Trajectory consists of series game step point tuples of:
/// + __state__ in the time just before __agent__ performing __action__,
/// + __agent__ that performed action,
/// + __action__ being performed,
/// + boolean evaluation of __action__ being legal (it usually should be legal, however in case it isn't
///     it may be noted here.
///
/// Trajectory ends with final state.
/// Data can be considered with following array:
/// | State | Agent | Action | valid |
/// |:-----:|:-----:|:------:|:-------|
/// | s\_0 | A | a\_0 | true |
/// | s\_1 | B | a\_1 | true |
/// | ... | ... | ... | true |
/// | s\_n | B| a\_n | true |
/// | final state | --- | --- | --- |
///
/// /// Each row is added via function [`GameTrajectory::register_step_point`] and the final row is added
/// via [`GameTrajectory::finish`].
///
/// For trace collected by players refer to [`AgentTrajectory`](crate::agent::AgentTrajectory).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct GameTrajectory<DP: Scheme, ST: SequentialGameState<DP>>{
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "ST: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "ST: serde::Serialize")))]
    states: Vec<ST>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::AgentId: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::AgentId: serde::Serialize")))]
    agents: Vec<DP::AgentId>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::ActionType: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::ActionType: serde::Serialize")))]
    actions: Vec<DP::ActionType>,
    validations: Vec<bool>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "ST: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "ST: serde::Serialize")))]
    final_state: Option<ST>,
}
impl<DP: Scheme, ST: SequentialGameState<DP>> Default for GameTrajectory<DP, ST>{
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
impl<DP: Scheme, ST: SequentialGameState<DP>> GameTrajectory<DP, ST>{

    /// Creates new trajectory. Initializing undergoing vectors with [`default`](Default::default).
    /// If size can be estimated before it will be better to use [`Self::with_capacity`]
    pub fn new() -> Self{
        Default::default()
    }
    /// Creates new trajectory with size hint to allocate memory at initialization.
    pub fn with_capacity(size: usize) -> Self{
        Self{
            states: Vec::with_capacity(size),
            agents: Vec::with_capacity(size),
            actions: Vec::with_capacity(size),
            validations: Vec::with_capacity(size),
            final_state: None
        }
    }
    /// Clears vectors of data. Uses [`Vec::clear`], so elements are removed but no reallocation occurs.
    pub fn clear(&mut self){
        self.actions.clear();
        self.states.clear();
        self.actions.clear();
        self.final_state = None;
    }

    /// Number of full steps (available for view). Step is complete when there is next information set
    /// and payoff able to select. This means that initially trajectory has step len 0, after
    /// first registration step len is still 0, as only initial information set is known and
    /// not information set at the end of step. Subsequent updates increment number of steps and finishing also.
    ///
    /// # Example:
    /// ```
    /// use amfiteatr_core::agent::AgentTrajectory;
    /// use amfiteatr_core::comm::StdEnvironmentEndpoint;
    /// use amfiteatr_core::demo::{DemoAction, DemoDomain, DemoInfoSet, DemoState};
    /// use amfiteatr_core::env::GameTrajectory;
    /// let state = DemoState::new_with_players(vec![(2.0, 2.0), (3.0, 3.0)], 1, &Default::default());
    /// let mut trajectory: GameTrajectory<DemoDomain, DemoState> = GameTrajectory::new();
    /// assert_eq!(trajectory.number_of_steps(), 0);
    /// trajectory.register_step_point(state.clone(), 0, DemoAction(0), true).unwrap();
    /// assert_eq!(trajectory.number_of_steps(), 0);
    /// trajectory.register_step_point(state.clone(), 0, DemoAction(1), true).unwrap();
    /// assert_eq!(trajectory.number_of_steps(), 1);
    /// trajectory.finish(state.clone()).unwrap();
    /// assert_eq!(trajectory.number_of_steps(), 2);
    ///
    /// ```
    pub fn number_of_steps(&self) -> usize{
        match self.is_finished(){
            true => self.states.len(),
            false => self.states.len().saturating_sub(1)
        }
    }
    /// Returns the number of raw action points registered. For finished trajectory it equals number
    /// of steps in game. For unfinished games, the number of steps is one less, as the noted step point
    /// has no follower.
    /// # Example:
    /// ```
    /// use amfiteatr_core::agent::AgentTrajectory;
    /// use amfiteatr_core::comm::StdEnvironmentEndpoint;
    /// use amfiteatr_core::demo::{DemoAction, DemoDomain, DemoInfoSet, DemoState};
    /// use amfiteatr_core::env::GameTrajectory;
    /// let state = DemoState::new_with_players(vec![(2.0, 2.0), (3.0, 3.0)], 1, &Default::default());
    /// let mut trajectory: GameTrajectory<DemoDomain, DemoState> = GameTrajectory::new();
    /// assert_eq!(trajectory.number_of_action_points(), 0);
    /// trajectory.register_step_point(state.clone(), 0, DemoAction(0), true).unwrap();
    /// assert_eq!(trajectory.number_of_action_points(), 1);
    /// trajectory.register_step_point(state.clone(), 0, DemoAction(1), true).unwrap();
    /// assert_eq!(trajectory.number_of_action_points(), 2);
    /// trajectory.finish(state.clone()).unwrap();
    /// assert_eq!(trajectory.number_of_action_points(), 2);
    pub fn number_of_action_points(&self) -> usize{
        self.states.len()
    }
    /// Register parameters for step. Provide starting state, action and payoff at start.
    pub fn register_step_point(&mut self, state: ST, agent: DP::AgentId, action: DP::ActionType, is_valid: bool)
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
    /// Registers state set and payoff in trajectory. Marks Trajectory as finished
    /// and no later [`Self::register_step_point`] and [`Self::finish`] are allowed.
    pub fn finish(&mut self, state: ST) -> Result<(), AmfiteatrError<DP>>{
        if self.final_state.is_some(){
            return Err(TrajectoryError::FinishingOnFinishedGameTrajectory {
                description: format!("{:?}", &self)}.into())
        }
        self.final_state = Some(state);
        Ok(())
    }
    /// Checks if trajectory is finished (the final game point is registered via method [`Self::finish`].
    pub fn is_finished(&self) -> bool{
        self.final_state.is_some()
    }
    /// Checks if there is no step points registered.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty() || self.validations.is_empty() || self.agents.is_empty() || self.actions.is_empty()
    }
    /// Returns view for last step in trajectory - if the game is not finished it will be
    /// step measured between two last registered entries. If game is finished (trajectory invoked
    /// function [`GameTrajectory::finish`]) then the step between last registered information set
    /// and information set provided for the finish function is used to mark last step.
    pub fn last_view_step(&self) -> Option<GameStepView<DP, ST>>{
        if self.final_state.is_some(){
            self.view_step(self.states.len().saturating_sub(1))
        } else {
            self.view_step(self.states.len().saturating_sub(2))
        }
    }
    /// Returns view of indexed step
    pub fn view_step(&self, index: usize) -> Option<GameStepView<DP, ST>>{
        self.states.get(index+1).map(|se|{
            GameStepView::new(
                &self.states[index],
                &self.agents[index],
                &self.actions[index],
                self.validations[index],
                se)
        }).or_else(||{
            if index + 1 == self.states.len(){
                /*
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

                 */
                self.final_state.as_ref().map(|fi_state| GameStepView::new(
                    &self.states[index],
                    &self.agents[index],
                    &self.actions[index],
                    self.validations[index],
                    fi_state
                ))
            } else { None}
        })
    }
}