use std::fmt::{Debug};
use crate::agent::InformationSet;
use crate::scheme::{Scheme, Reward};
use crate::error::AmfiteatrError;
use crate::error::TrajectoryError::UpdateOnFinishedAgentTrajectory;



/// View of single step in game trajectory.
///
/// Provides references to information sets and
/// scheme payoffs both for time moment before taking action and at the moment before player does next
/// game action.
/// View contains also action made in this step. Reward for step is calculated as difference in
/// payoffs. It may be also calculated as difference in assessments provided by InformationSet
/// implementing [`EvaluatedInformationSet`](crate::agent::EvaluatedInformationSet).
///
///
/// __Note__: Unlike the [`GameStepView`](crate::env::GameStepView),  step is measured from the moment of taking action to the moment just before next
/// action of the same player.
/// For one [`AgentStepView`] there might be  one or more [`GameStepView`](crate::env::GameStepView)s registered by the environment.
/// In the meantime action multiple updates on information set and payoff can be made.
///
/// For trace collected by central environment refer to [`GameTrajectory`](crate::agent::AgentTrajectory).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct AgentStepView<'a, DP: Scheme, IS: InformationSet<DP>>{

    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a IS: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a IS: serde::Serialize")))]
    pub(crate) start_info_set: &'a IS,
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a DP::UniversalReward: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a DP::UniversalReward: serde::Deserialize<'de>")))]
    pub(crate) start_payoff: &'a DP::UniversalReward,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a IS: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a IS: serde::Serialize")))]
    pub(crate) end_info_set: &'a IS,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a DP::UniversalReward: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a DP::UniversalReward: serde::Serialize")))]
    pub(crate) end_payoff: &'a DP::UniversalReward,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "&'a DP::ActionType: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "&'a DP::ActionType: serde::Serialize")))]
    pub(crate) action: &'a DP::ActionType,
    pub(crate) legality: bool

}

impl<'a, DP: Scheme, IS: InformationSet<DP>> AgentStepView<'a, DP, IS>{
    pub fn new(start_info_set: &'a IS, end_info_set: &'a IS,
               start_payoff: &'a DP::UniversalReward, end_payoff: &'a DP::UniversalReward,
    action: &'a DP::ActionType, legality: bool) -> Self{
        Self{
            start_info_set,
            start_payoff,
            end_info_set,
            end_payoff,
            action,
            legality
        }
    }
    /// Information set before taking action.
    pub fn information_set(&self) -> &IS{
        self.start_info_set
    }
    /// Payoff set before taking action.
    pub fn payoff(&self) -> &DP::UniversalReward{
        self.start_payoff
    }
    /// Information set after step completion (just before next step).
    pub fn late_information_set(&self) -> &IS{
        self.end_info_set
    }
    /// Payoff after step completion (just before next step).
    pub fn late_payoff(&self) -> &DP::UniversalReward{
        self.end_payoff
    }
    /// Difference in late payoff and start payoff calculated via [`Reward::ref_sub`].
    pub fn reward(&self) -> DP::UniversalReward{
        self.end_payoff.ref_sub(self.start_payoff)
    }
    /// Action taken in this step.
    pub fn action(&self) -> &DP::ActionType{
        self.action
    }

    /// If action is legal returns `true`, else `false`. It says about `action` in `information_set`,
    /// does not say anything about step initial information set, but says that next information set
    /// is to be considered illegal..
    pub fn is_action_legal(&self) -> bool{
        self.legality
    }

}


/// Whole trajectory structure from the point of view of agent.
/// Consists of sequences of [`InformationSet`]s, respectful payoffs [`Reward`].
///
///
/// Trajectory consists of series step points tuples of:
/// +  __information set__ in time point of performing action,
/// + the __action__ selected,
/// + __payoff__ in the moment of performing action.
///
/// Trajectory ends with final information set (state) and corresponding payoff (no action performed there).
/// Data can be considered with following array:
///
///
/// | Information Set | Action | Payoff |
/// |:---------------:|:------:|:-------|
/// | s\_0 | a\_0 | p\_0 |
/// | s\_1 | a\_1 | p\_1 |
/// | ... | ... | ... |
/// | s\_n | a\_n | p\_n |
/// | final state | --- | final payoff |
///
/// Each row is added via function [`AgentTrajectory::register_step_point`] and the final row is added
/// via [`AgentTrajectory::finish`].
///
/// One game step is measured between point in time for one performed action and next action performed.
/// Note that during single step information set and payoff may be updated one or more times as other players
/// may change the game state in the meantime.
/// It is important to catch game information at the exact moments of taking actions.
/// The reward for _a\_0_ in previous table is _p\_1 - p\_0_, therefore to calculate reward for game step
/// it is needed that the initial point is known and next point - this may be begining of next step or
/// final point in game.
///
/// Similarly, agent side calculated assessment on information set implementing [`EvaluatedInformationSet`](crate::agent::EvaluatedInformationSet)
/// can be done by subtracting [`current_assessment`](crate::agent::EvaluatedInformationSet::current_assessment).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct AgentTrajectory<DP: Scheme, IS: InformationSet<DP>>{

    #[cfg_attr(feature = "serde", serde(bound(deserialize = "IS: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "IS: serde::Serialize")))]
    information_sets: Vec<IS>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::UniversalReward: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::UniversalReward: serde::Serialize")))]
    payoffs: Vec<DP::UniversalReward>,
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::ActionType: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::ActionType: serde::Serialize")))]
    actions: Vec<DP::ActionType>,

    #[cfg_attr(feature = "serde", serde(bound(deserialize = "IS: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "IS: serde::Serialize")))]
    final_info_set: Option<IS>,

    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::UniversalReward: serde::Deserialize<'de>")))]
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::UniversalReward: serde::Serialize")))]
    final_payoff: Option<DP::UniversalReward>,

    legalities: Vec<bool>,

    //illegal_combinations: Vec<S,DP::ActionType>


}



impl<DP: Scheme, IS: InformationSet<DP>>  Default for AgentTrajectory<DP, IS>{
    fn default() -> Self {
        Self{ information_sets: vec![], payoffs: vec![], actions: vec![],
            final_info_set: None, final_payoff: None, legalities: vec![],
            //illegal_combinations: vec![]
        }
    }
}

impl<DP: Scheme, IS: InformationSet<DP>> AgentTrajectory<DP, IS>{

    /// Creates new trajectory. Initializing undergoing vectors with [`default`](Default::default).
    /// If size can be estimated before it will be better to use [`Self::with_capacity`]
    pub fn new() -> Self{
        Default::default()
    }

    /// Clears vectors of data. Uses [`Vec::clear`], so elements are removed but no reallocation occurs.
    pub fn clear(&mut self){
        self.actions.clear();
        self.payoffs.clear();
        self.information_sets.clear();
        self.final_payoff = None;
        self.final_info_set = None;
        //self.illegal_combinations.clear();
    }

    /// Creates new trajectory with size hint to allocate memory at initialization.
    pub fn with_capacity(size: usize) -> Self{
        Self{
            information_sets: Vec::with_capacity(size),
            payoffs: Vec::with_capacity(size),
            actions: Vec::with_capacity(size),
            final_info_set: None,
            final_payoff: None,
            legalities: Vec::with_capacity(size),
            //illegal_combinations: Vec::new()
        }
    }

    /// Register parameters for step. Provide starting information set, action and payoff at start.
    pub fn register_step_point(&mut self, info_set: IS, action: DP::ActionType, payoff: DP::UniversalReward)
                               -> Result<(), AmfiteatrError<DP>>{
        if self.final_payoff.is_some() || self.final_info_set.is_some(){
            return Err(AmfiteatrError::Trajectory{ source: UpdateOnFinishedAgentTrajectory(info_set.agent_id().clone())})
        }
        self.payoffs.push(payoff);
        self.actions.push(action);
        self.information_sets.push(info_set);
        self.legalities.push(true);
        Ok(())
    }

    /// Marks previously committed step point as illegal
    pub fn mark_previous_action_illegal(&mut self){
        if let Some(valid) = self.legalities.last_mut(){
            *valid = false;
        }
    }


    

    /// Registers final information set and payoff in trajectory. Marks Trajectory as finished
    /// and no later [`Self::register_step_point`] and [`Self::finish`] are allowed.
    pub fn finish(&mut self, info_set: IS,  payoff: DP::UniversalReward) -> Result<(), AmfiteatrError<DP>>{
        if self.final_payoff.is_some() || self.final_info_set.is_some(){
            return Err(AmfiteatrError::Trajectory{ source: UpdateOnFinishedAgentTrajectory(info_set.agent_id().clone())})
        }
        self.final_info_set = Some(info_set);
        self.final_payoff = Some(payoff);
        Ok(())
    }

    /// Checks if trajectory is finished (the final game point is registered via method [`Self::finish`].
    pub fn is_finished(&self) -> bool{
        self.final_payoff.is_some() && self.final_info_set.is_some()
    }

    /// Checks if there is no step points registered.
    pub fn is_empty(&self) -> bool{
        self.information_sets.is_empty() || self.actions.is_empty() || self.payoffs.is_empty()
    }

    /// Returns view for last step in trajectory - if the game is not finished it will be
    /// step measured between two last registered entries. If game is finished (trajectory invoked
    /// function [`AgentTrajectory::finish`]) then the step between last registered information set
    /// and information set provided for the finish function is used to mark last step.
    pub fn last_view_step(&self) -> Option<AgentStepView<DP, IS>>{
        if self.final_payoff.is_some() && self.final_info_set.is_some(){
            self.view_step(self.payoffs.len().saturating_sub(1))
        } else {
            self.view_step(self.payoffs.len().saturating_sub(2))
        }
    }

    /// Returns view of indexed step
    pub fn view_step(&self, index: usize) -> Option<AgentStepView<DP, IS>>{
        //first check if there is next normal step
        self.information_sets.get(index+1).and_then(|se|{
            self.payoffs.get(index+1).map(|pe|{
                AgentStepView::new(
                    &self.information_sets[index],
                    se, &self.payoffs[index],
                    pe,
                    &self.actions[index],
                    self.legalities[index],
                )
            })
        }).or_else(||{
            //check if we ask for last data in normal vector
            if index + 1 == self.information_sets.len(){
                // check if step is final
                if let (Some(info_set), Some(payoff)) = (&self.final_info_set, &self.final_payoff){
                    Some(AgentStepView::new(
                        &self.information_sets[index],
                        info_set,
                        &self.payoffs[index],
                        payoff,
                        &self.actions[index],
                        self.legalities[index]
                    ))
                } else {
                    None
                }
            } else {
                None
            }

        })


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
    /// use amfiteatr_core::demo::{DemoAction, DemoDomain, DemoInfoSet};
    /// let infoset = DemoInfoSet::new(1, 2);
    /// let mut trajectory: AgentTrajectory<DemoDomain, DemoInfoSet> = AgentTrajectory::new();
    /// assert_eq!(trajectory.number_of_steps(), 0);
    /// trajectory.register_step_point(infoset.clone(), DemoAction(0), 0.0).unwrap();
    /// assert_eq!(trajectory.number_of_steps(), 0);
    /// trajectory.register_step_point(infoset.clone(), DemoAction(1), 2.0).unwrap();
    /// assert_eq!(trajectory.number_of_steps(), 1);
    /// trajectory.finish(infoset.clone(), 3.0).unwrap();
    /// assert_eq!(trajectory.number_of_steps(), 2);
    ///
    /// ```
    pub fn number_of_steps(&self) -> usize{
        match self.is_finished(){
            true => self.information_sets.len(),
            false => self.information_sets.len().saturating_sub(1),
        }

    }
    /// Returns the number of raw action points registered. For finished trajectory it equals number
    /// of steps in game. For unfinished games, the number of steps is one less, as the noted step point
    /// has no follower.
    /// # Example:
    /// ```
    /// use amfiteatr_core::agent::AgentTrajectory;
    /// use amfiteatr_core::comm::StdEnvironmentEndpoint;
    /// use amfiteatr_core::demo::{DemoAction, DemoDomain, DemoInfoSet};
    /// let infoset = DemoInfoSet::new(1, 2);
    /// let mut trajectory: AgentTrajectory<DemoDomain, DemoInfoSet> = AgentTrajectory::new();
    /// assert_eq!(trajectory.number_of_action_points(), 0);
    /// trajectory.register_step_point(infoset.clone(), DemoAction(0), 0.0).unwrap();
    /// assert_eq!(trajectory.number_of_action_points(), 1);
    /// trajectory.register_step_point(infoset.clone(), DemoAction(1), 2.0).unwrap();
    /// assert_eq!(trajectory.number_of_action_points(), 2);
    /// trajectory.finish(infoset.clone(), 3.0).unwrap();
    /// assert_eq!(trajectory.number_of_action_points(), 2);
    ///
    /// ```
    pub fn number_of_action_points(&self) -> usize{
        self.information_sets.len()
    }

    /// Returns iterator of step views.
    pub fn iter(&self) -> AgentStepIterator<DP, IS>{
        AgentStepIterator{
            trajectory: self,
            index: 0,
        }
    }

    /// From a slice of trajectories finds the length of the longest one.
    #[inline]
    pub fn find_max_trajectory_len
    (
        trajectories: &[AgentTrajectory<DP, IS>]
    ) -> usize{
        trajectories.iter().map(|x|{
            x.number_of_steps()
        }).max().unwrap_or(0)
    }

    /// Sum the number of steps in slice of trajectories.
    #[inline]
    pub fn sum_trajectories_steps
    (
        trajectories: &[AgentTrajectory<DP, IS>]
    ) -> usize{
        trajectories.iter().fold(0, |acc, x|{
            acc + x.number_of_steps()
        })
    }


}

/// Iterator for step views in agent trajectory
pub struct AgentStepIterator<'a, DP: Scheme, IS: InformationSet<DP>>{
    trajectory: &'a AgentTrajectory<DP, IS>,
    index: usize,
}


impl<'a, DP: Scheme, IS: InformationSet<DP>> Iterator for AgentStepIterator<'a, DP, IS>{
    type Item = AgentStepView<'a, DP, IS>;

    fn next(&mut self) -> Option<Self::Item> {

        self.trajectory.view_step(self.index).inspect(|_v|{
            self.index += 1;
        })
    }
}

impl<DP: Scheme, IS: InformationSet<DP>> ExactSizeIterator for AgentStepIterator<'_, DP, IS>{
    fn len(&self) -> usize{
        match self.trajectory.final_info_set{
            Some(_) => self.trajectory.information_sets.len(),
            None => self.trajectory.information_sets.len().saturating_sub(1)
        }
    }
}


#[cfg(test)]
mod tests{
    use std::collections::HashMap;
    use std::thread;
    use crate::agent::{AutomaticAgent, RandomPolicy, TracingAgent, TracingAgentGen};
    use crate::comm::StdEnvironmentEndpoint;
    use crate::demo::{DEMO_AGENT_BLUE, DEMO_AGENT_RED, DemoDomain, DemoInfoSet, DemoPolicySelectFirst, DemoState};
    use crate::env::{RoundRobinUniversalEnvironment, TracingHashMapEnvironment};

    #[test]
    fn agent_trajectory_test(){
        let bandits = vec![(5.0,5.0), (11.5, 11.5), (6.0, 6.0)];
        let number_of_bandits = bandits.len();
        let (comm_env_blue, comm_agent_blue) = StdEnvironmentEndpoint::new_pair();
        let (comm_env_red, comm_agent_red) = StdEnvironmentEndpoint::new_pair();
        let mut env_comms = HashMap::new();
        env_comms.insert(DEMO_AGENT_BLUE, comm_env_blue);
        env_comms.insert(DEMO_AGENT_RED, comm_env_red);
        let player_set = env_comms.keys().map(|id| *id).collect();
        let state = DemoState::new_with_players(bandits, 3, &player_set);
        let mut environment = TracingHashMapEnvironment::new(state, env_comms);
        let blue_info_set = DemoInfoSet::new(DEMO_AGENT_BLUE, number_of_bandits);
        let red_info_set = DemoInfoSet::new(DEMO_AGENT_RED, number_of_bandits);
        let mut agent_blue = TracingAgentGen::new(blue_info_set, comm_agent_blue,  RandomPolicy::<DemoDomain, DemoInfoSet>::new());
        let mut agent_red = TracingAgentGen::new(red_info_set, comm_agent_red, DemoPolicySelectFirst{});
        thread::scope(|s|{
            s.spawn(||{
                environment.run_round_robin_with_rewards().unwrap();
            });
            s.spawn(||{
                agent_blue.run().unwrap();
            });
            s.spawn(||{
                agent_red.run().unwrap();
            });
        });



        assert!(agent_blue.trajectory().is_finished());
        assert_eq!(agent_blue.trajectory().information_sets.len(), 3);
        assert_eq!(agent_blue.trajectory().number_of_steps(), 3);
        assert_eq!(agent_red.trajectory().view_step(0).unwrap().payoff().clone(), 0.0);

        assert_eq!(agent_red.trajectory().view_step(0).unwrap().late_payoff(), &5.0);
        assert_eq!(agent_red.trajectory().view_step(1).unwrap().payoff(), &5.0);
        assert_eq!(agent_red.trajectory().view_step(1).unwrap().late_payoff(), &10.0);
        assert_eq!(agent_red.trajectory().view_step(2).unwrap().payoff(), &10.0);
        assert_eq!(agent_red.trajectory().view_step(2).unwrap().late_payoff(), &15.0);



        assert!(agent_blue.trajectory().view_step(0).unwrap().payoff() < agent_blue.trajectory().view_step(0).unwrap().late_payoff());
        assert_eq!(agent_blue.trajectory().view_step(0).unwrap().late_payoff(), agent_blue.trajectory().view_step(1).unwrap().payoff());
        assert!(agent_blue.trajectory().view_step(1).unwrap().late_payoff() < agent_blue.trajectory().view_step(2).unwrap().late_payoff());

        assert_eq!(agent_blue.trajectory().iter().len(), 3);
    }
}