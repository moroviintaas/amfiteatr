use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Sub};


/// `Reward` is trait for types to be used as policy performance assessment.
/// It is implemented for standard types and you can use your own type as reward,
/// if only you made it partially comparable and summable.
pub trait Reward: Send + Clone + Debug + PartialEq  + PartialOrd + Default +
    for<'a> Add<&'a Self, Output=Self> + Add<Output=Self> + Add + for<'a> AddAssign<&'a Self>
    + Sub<Output=Self> + for<'a> Sub<&'a Self, Output=Self> + Sub {
    /// This is constructor used to produce neutral value of reward, i.e.
    /// the reward that does not change the score. For standard numeric
    /// types this is just value of 0.
    fn neutral() -> Self;
    
    fn ref_sub(&self, rhs: &Self) -> Self;

}



/// Reward that can be compared to another with proportion (division) resulting in float
pub trait ProportionalReward<Float>: Reward{
    fn proportion(&self, other: &Self) -> Float;
}


macro_rules! impl_reward_std {
    ($($x: ty), +) => {
        $(
          impl Reward for $x{
              fn neutral() -> $x{
                  0
              }
              fn ref_sub(&self, rhs: &Self) -> Self {
                  self - rhs
            }
          }

        )*

    }
}

impl_reward_std![u8, u16, u32, u64, i8, i16, i32, i64];

impl Reward for f32{
    fn neutral() -> Self {
        0.0
    }

    fn ref_sub(&self, rhs: &Self) -> Self {
        self - rhs
    }
}
impl Reward for f64{
    fn neutral() -> Self {
        0.0
    }

    fn ref_sub(&self, rhs: &Self) -> Self {
        self - rhs
    }
}


/*
#[derive(Debug, Copy, Clone)]
pub enum RewardSource{
    Env,
    Agent
}

 */

/*
impl ProportionalReward<f32> for f32{}
impl ProportionalReward<f64> for f64{}
impl ProportionalReward<f32> for i64{}
 */

impl ProportionalReward<f32> for f32{
    fn proportion(&self, other: &Self) -> f32 {
        self / other
    }
}

impl ProportionalReward<f32> for i32{
    fn proportion(&self, other: &Self) -> f32 {
        *self as f32/ *other as f32
    }
}


/// Reward of none type, use if reward is irrelevant because some traits expects that reward is defined.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct NoneReward{}



impl PartialOrd for NoneReward {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}


impl<'a> Add<&'a Self> for NoneReward {
    type Output = NoneReward;

    fn add(self, _: &'a Self) -> Self::Output {
        NoneReward{}
    }
}

impl Add for NoneReward {
    type Output = NoneReward;

    fn add(self, _rhs: Self) -> Self::Output {
        NoneReward{}
    }
}


impl<'a> AddAssign<&'a Self> for NoneReward {
    fn add_assign(&mut self, _rhs: &'a Self) {

    }
}

impl Sub for NoneReward {
    type Output = NoneReward;

    fn sub(self, _rhs: Self) -> Self::Output {
        NoneReward{}
    }
}

impl<'a> Sub<&'a Self> for NoneReward {
    type Output = NoneReward;

    fn sub(self, _rhs: &'a Self) -> Self::Output {
        NoneReward{}
    }
}


impl Reward for NoneReward{
    fn neutral() -> Self {
        NoneReward{}
    }

    fn ref_sub(&self, _rhs: &Self) -> Self {
        Self::neutral()
    }
}