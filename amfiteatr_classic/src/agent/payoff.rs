use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Index, IndexMut, Sub};
use enum_map::{enum_map, EnumMap};
use serde::{Serialize};
use amfiteatr_core::domain::Reward;
use crate::domain::{ClassicAction, IntReward};
use crate::domain::ClassicAction::{Down, Up};


/// [`EnumMap`] mapping generic type to [`ClassicAction`] used here to store for examples counts
/// of actions made.
pub type Level1ActionMap<T> = EnumMap<ClassicAction, T>;
/// [`EnumMap`] mapping [`Level1ActionMap`] to [`ClassicAction`] used here to store for examples counts
/// of actions made. You probably want to use wrapping structure [`ActionPairMapper`].

pub type Level2ActionMap<T> = EnumMap<ClassicAction, Level1ActionMap<T>>;


/// Structure to map some data to action pair (for example count number of situation where player 1
/// used action A, and player 2 used action B)
/// ```
/// use amfiteatr_classic::agent::ActionPairMapper;
/// use amfiteatr_classic::domain::ClassicAction::{Up, Down};
/// let mut mapper = ActionPairMapper::zero();
/// assert_eq!(mapper[Up][Down], 0);
/// mapper[Up][Down] = 7i64;
/// assert_eq!(mapper[Up][Down], 7);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct ActionPairMapper<T: Copy + Clone + Debug + PartialEq>(Level2ActionMap<T>);

impl<T: Copy + Clone + Debug + PartialEq> ActionPairMapper<T>{
    pub fn new(map: Level2ActionMap<T>) -> Self{
        Self(map)
    }
}
/*
impl<T: Copy + Clone + Debug + PartialEq + Serialize> Serialize for ActionPairMapper<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let mut state = serializer.serialize_struct("ActionPairMapper", 1)?;
        state.serialize_field("map", &self.0)?;
        state.end()
    }
}

 */

impl ActionPairMapper<i64>{
    pub fn zero() -> Self{
        Self::default()
    }
}

impl<T: Copy + Clone + Debug + PartialEq> Index<ClassicAction> for ActionPairMapper<T>{
    type Output = Level1ActionMap<T>;

    fn index(&self, index: ClassicAction) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Copy + Clone + Debug + PartialEq> IndexMut<ClassicAction> for ActionPairMapper<T>{

    fn index_mut(&mut self, index: ClassicAction) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Copy + Clone + Debug + PartialEq + Default> Default for ActionPairMapper<T>{
    fn default() -> Self {
        Self(enum_map! {
            ClassicAction::Up => enum_map! {
                ClassicAction::Up => T::default(),
                ClassicAction::Down => T::default()
            },
            ClassicAction::Down => enum_map! {
                ClassicAction::Up => T::default(),
                ClassicAction::Down => T::default()
            }
        })
    }
}

impl<T: Copy + Clone + Debug + Add<Output = T> + PartialEq> Add for ActionPairMapper<T>{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ActionPairMapper::new(enum_map! {
            ClassicAction::Up => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Up][ClassicAction::Up]
                    + rhs[ClassicAction::Up][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Up][ClassicAction::Down]
                    + rhs[ClassicAction::Up][ClassicAction::Down],
            },
            ClassicAction::Down => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Down][ClassicAction::Up]
                    + rhs[ClassicAction::Down][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Down][ClassicAction::Down]
                    + rhs[ClassicAction::Down][ClassicAction::Down],
            },
        })
    }
}

impl<'a, T: Copy + Clone + Debug + Add<Output = T> + PartialEq> Add<&'a Self> for ActionPairMapper<T>{
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        ActionPairMapper(enum_map! {
            ClassicAction::Up => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Up][ClassicAction::Up]
                    + rhs[ClassicAction::Up][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Up][ClassicAction::Down]
                    + rhs[ClassicAction::Up][ClassicAction::Down],
            },
            ClassicAction::Down => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Down][ClassicAction::Up]
                    + rhs[ClassicAction::Down][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Down][ClassicAction::Down]
                    + rhs[ClassicAction::Down][ClassicAction::Down],
            },
        })
    }
}

impl<'a, T: Copy + Clone + Debug + AddAssign + PartialEq> AddAssign<&'a Self> for ActionPairMapper<T>{

    fn add_assign(&mut self, rhs: &'a Self){
        self[Down][Down] += rhs[Down][Down];
        self[Down][Up] += rhs[Down][Up];
        self[Up][Down] += rhs[Up][Down];
        self[Up][Up] += rhs[Up][Up];

        /*
        ActionCounter(enum_map! {
            ClassicAction::Defect => enum_map! {
                ClassicAction::Defect =>  self[ClassicAction::Defect][ClassicAction::Defect]
                    + rhs[ClassicAction::Defect][ClassicAction::Defect],
                ClassicAction::Cooperate =>  self[ClassicAction::Defect][ClassicAction::Cooperate]
                    + rhs[ClassicAction::Defect][ClassicAction::Cooperate],
            },
            ClassicAction::Cooperate => enum_map! {
                ClassicAction::Defect =>  self[ClassicAction::Cooperate][ClassicAction::Defect]
                    + rhs[ClassicAction::Cooperate][ClassicAction::Defect],
                ClassicAction::Cooperate =>  self[ClassicAction::Cooperate][ClassicAction::Cooperate]
                    + rhs[ClassicAction::Cooperate][ClassicAction::Cooperate],
            },
        })

         */
    }
}

impl<T: Copy + Clone + Debug + Sub<Output = T> + PartialEq> Sub for ActionPairMapper<T>{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        ActionPairMapper::new(enum_map! {
            ClassicAction::Up => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Up][ClassicAction::Up]
                    - rhs[ClassicAction::Up][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Up][ClassicAction::Down]
                    - rhs[ClassicAction::Up][ClassicAction::Down],
            },
            ClassicAction::Down => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Down][ClassicAction::Up]
                    - rhs[ClassicAction::Down][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Down][ClassicAction::Down]
                    - rhs[ClassicAction::Down][ClassicAction::Down],
            },
        })
    }
}

impl<'a, T: Copy + Clone + Debug + Sub<Output = T> + PartialEq> Sub<&'a Self> for ActionPairMapper<T>{
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        ActionPairMapper(enum_map! {
            ClassicAction::Up => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Up][ClassicAction::Up]
                    - rhs[ClassicAction::Up][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Up][ClassicAction::Down]
                    - rhs[ClassicAction::Up][ClassicAction::Down],
            },
            ClassicAction::Down => enum_map! {
                ClassicAction::Up =>  self[ClassicAction::Down][ClassicAction::Up]
                    - rhs[ClassicAction::Down][ClassicAction::Up],
                ClassicAction::Down =>  self[ClassicAction::Down][ClassicAction::Down]
                    - rhs[ClassicAction::Down][ClassicAction::Down],
            },
        })
    }
}


/// Sample complex reward to demonstrate approach to multi-objective optimization.
/// Among standard table derived (game defined) payoff, this struct provides counting of
/// actions made so you can make optimization criteria to for example maximize number of encounters
/// where both players played [`Down`](ClassicAction::Down).
/// ```
/// use amfiteatr_classic::agent::AgentAssessmentClassic;
/// use amfiteatr_classic::domain::ClassicAction::Down;
/// let preference_model: Box< dyn Fn(AgentAssessmentClassic<i64>) -> f32> = Box::new(| reward |{
///     reward.table_payoff() as f32 + (0.4* reward.count_actions(Down, Down) as f32)
/// });
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Default, Serialize)]
pub struct AgentAssessmentClassic<R: Reward + Copy>{
    table_payoff: R,
    //count_coop_vs_coop: i64,
    //count_coop_vs_defect: i64,
    //count_defect_vs_coop: i64,
    //count_defect_vs_defect: i64,
    action_counts: ActionPairMapper<i64>,
    education_assessment: f32,

}


impl<R: Reward + Copy> PartialOrd for AgentAssessmentClassic<R> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.table_payoff.partial_cmp(&other.table_payoff)
    }
}

impl<R: Reward + Copy> AgentAssessmentClassic<R>{

    pub fn new(table_payoff: R, action_counts: ActionPairMapper<i64>, education_assessment: f32) -> Self{
        Self{table_payoff, action_counts, education_assessment }
    }

    pub fn with_only_table_payoff(payoff: R) -> Self{
        Self{
            table_payoff: payoff,
            action_counts: ActionPairMapper::zero(),
            education_assessment: 0.0,
        }
    }
}

impl AgentAssessmentClassic<IntReward>{

    pub fn table_payoff(&self) -> IntReward{
        self.table_payoff
    }

    pub fn count_own_actions(&self, action: ClassicAction) -> IntReward{
        match action{
            Up => self.action_counts[Up][Down] + self.action_counts[Up][Up],
            Down => self.action_counts[Down][Down] + self.action_counts[Down][Up],
        }
    }

    pub fn count_other_actions(&self, action: ClassicAction) -> IntReward{
        match action{
            Up => self.action_counts[Down][Up] + self.action_counts[Up][Up],
            Down => self.action_counts[Down][Down] + self.action_counts[Up][Down],
        }
    }

    pub fn count_actions(&self, own: ClassicAction, other: ClassicAction) -> i64{
        self.action_counts[own][other]
    }

    pub fn other_coop_as_reward(&self) -> IntReward{
        self.action_counts[Down][Down] + self.action_counts[Up][Down]
    }

    pub fn coops_as_reward(&self) -> IntReward{
        3 * (self.action_counts[Down][Down] + self.action_counts[Up][Down])
         + self.action_counts[Down][Up]
    }

    pub fn f_combine_table_with_other_coop(&self, action_count_weight: f32) -> f32{
        self.table_payoff as f32 + (action_count_weight * self.count_other_actions(Down) as f32)
    }

    pub fn count_both_actions(&self, action: ClassicAction) -> IntReward{
        self.count_other_actions(action) + self.count_own_actions(action)
    }
    pub fn f_combine_table_with_both_coop(&self, action_count_weight: f32) -> f32{
        self.table_payoff as f32 + (action_count_weight * self.count_both_actions(Down) as f32)
    }

    pub fn education_assessment(&self) -> f32{
        self.education_assessment
    }

    pub fn combine_edu_assessment(&self, assessment_weight: f32) -> f32{
        assessment_weight * self.education_assessment + self.table_payoff as f32
    }



}



impl<'a, R: Reward + Copy> Add<&'a Self> for AgentAssessmentClassic<R> {
    type Output = AgentAssessmentClassic<R>;

    fn add(self, rhs: &'a Self) -> Self::Output {
        Self{
            table_payoff: self.table_payoff + rhs.table_payoff,
            /*
            count_coop_vs_coop: self.count_coop_vs_coop + rhs.count_coop_vs_coop,
            count_coop_vs_defect: self.count_coop_vs_defect + rhs.count_coop_vs_defect,
            count_defect_vs_coop: self.count_defect_vs_coop + rhs.count_defect_vs_coop,
            count_defect_vs_defect: self.count_defect_vs_defect + rhs.count_defect_vs_defect

             */
            action_counts: self.action_counts + rhs.action_counts,
            education_assessment: self.education_assessment + rhs.education_assessment,
        }
    }
}

impl<R: Reward + Copy> Add for AgentAssessmentClassic<R> {
    type Output = AgentAssessmentClassic<R>;

    fn add(self, rhs: Self) -> Self::Output {
        Self{
            table_payoff: self.table_payoff + rhs.table_payoff,
            action_counts: self.action_counts + rhs.action_counts,
            education_assessment: self.education_assessment + rhs.education_assessment
            /*
            count_coop_vs_coop: self.count_coop_vs_coop + rhs.count_coop_vs_coop,
            count_coop_vs_defect: self.count_coop_vs_defect + rhs.count_coop_vs_defect,
            count_defect_vs_coop: self.count_defect_vs_coop + rhs.count_defect_vs_coop,
            count_defect_vs_defect: self.count_defect_vs_defect + rhs.count_defect_vs_defect

             */
        }
    }
}



impl<'a, R: Reward + Copy> AddAssign<&'a Self> for AgentAssessmentClassic<R> {
    fn add_assign(&mut self, rhs: &'a Self) {
        self.table_payoff += &rhs.table_payoff;
        /*
        self.count_coop_vs_coop += &rhs.count_coop_vs_coop;
        self.count_coop_vs_defect += &rhs.count_coop_vs_defect;
        self.count_defect_vs_coop += &rhs.count_defect_vs_coop;
        self.count_defect_vs_defect += &rhs.count_defect_vs_defect;

         */

        self.action_counts += &rhs.action_counts;
        self.education_assessment += &rhs.education_assessment;
    }
}

impl<R: Reward + Copy> Sub for AgentAssessmentClassic<R> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self{
            table_payoff: self.table_payoff - rhs.table_payoff,
            /*
            count_coop_vs_coop: self.count_coop_vs_coop - rhs.count_coop_vs_coop,
            count_coop_vs_defect: self.count_coop_vs_defect - rhs.count_coop_vs_defect,
            count_defect_vs_coop: self.count_defect_vs_coop - rhs.count_defect_vs_coop,
            count_defect_vs_defect: self.count_defect_vs_defect - rhs.count_defect_vs_defect

             */
            action_counts : self.action_counts - rhs.action_counts,
            education_assessment: self.education_assessment - rhs.education_assessment
        }
    }
}

impl<'a, R: Reward + Copy> Sub<&'a Self> for AgentAssessmentClassic<R> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        Self{
            table_payoff: self.table_payoff - rhs.table_payoff,
            action_counts : self.action_counts - rhs.action_counts,
            education_assessment: self.education_assessment - rhs.education_assessment
        }
    }
}





impl<R: Reward + Copy> Reward for AgentAssessmentClassic<R>{
    fn neutral() -> Self {
        Self{
            table_payoff: R::neutral(),
            action_counts: ActionPairMapper::zero(),
            education_assessment: 0.0
        }
    }

    fn ref_sub(&self, rhs: &Self) -> Self {
        Self{
            table_payoff: self.table_payoff - rhs.table_payoff,
            action_counts : self.action_counts - rhs.action_counts,
            education_assessment: self.education_assessment - rhs.education_assessment
        }
    }
}

