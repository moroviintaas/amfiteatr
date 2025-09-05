use std::ops::Div;
use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use amfiteatr_core::agent::Policy;
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_rl::policy::PolicySpecimen;
use crate::agent::LocalHistoryInfoSet;
use crate::scheme::{ClassicAction, ClassicScheme, UsizeAgentId};

#[derive(Copy, Clone, Debug, Default)]
pub struct EventCounts {
    pub count_up_v_up: f64,
    pub count_up_v_down: f64,
    pub count_down_v_up: f64,
    pub count_down_v_down: f64,

    pub count_im_punished_immediately: f64,
    pub count_i_punish_immediately: f64,

    pub count_im_punished_after2: f64,
    pub count_i_punish_after2: f64,

    pub count_im_absoluted_immediately: f64,
    pub count_i_absolute_immediately: f64,




}

impl EventCounts{
    pub fn new() -> Self{
        Self::default()
    }
}

impl Div<f64> for &EventCounts{
    type Output = EventCounts;

    fn div(self, rhs: f64) -> Self::Output {
        EventCounts{
            count_up_v_up: self.count_up_v_up/rhs,
            count_up_v_down: self.count_up_v_down/rhs,
            count_down_v_up: self.count_down_v_up/rhs,
            count_down_v_down: self.count_down_v_down/rhs,
            count_im_punished_immediately: self.count_im_absoluted_immediately/rhs,
            count_i_punish_immediately: self.count_i_punish_immediately/rhs,
            count_im_punished_after2: self.count_im_punished_after2/rhs,
            count_i_punish_after2: self.count_i_punish_after2/rhs,
            count_im_absoluted_immediately: self.count_im_absoluted_immediately/rhs,
            count_i_absolute_immediately: self.count_i_absolute_immediately/rhs,
        }
    }
}


pub struct GpClassic{

    pub w_up_v_up: f64,
    pub w_down_v_up: f64,
    pub w_up_v_down: f64,
    pub w_down_v_down: f64,

    pub w_i_punish_1: f64,
    pub w_im_punished_1: f64,
    pub w_i_absolute_1: f64,
    pub w_im_absoluted_1: f64,

    pub w_i_punish_2: f64,
    pub w_im_punished_2: f64,

    pub stdev: f64

}

impl GpClassic{

    fn _policy_get_gauss_mean<ID: UsizeAgentId>(&self, information_set: &LocalHistoryInfoSet<ID>) -> f64{

        let probablities = information_set.calculate_past_event_probabilities();

        (probablities.count_up_v_up * self.w_up_v_up) +
            (probablities.count_up_v_down * self.w_up_v_down) +
            (probablities.count_down_v_up * self.w_down_v_up) +
            (probablities.count_down_v_down * self.w_down_v_down) +
            (probablities.count_i_punish_immediately * self.w_i_punish_1 ) +
            (probablities.count_im_punished_immediately * self.w_im_punished_1 ) +
            (probablities.count_i_absolute_immediately * self.w_i_absolute_1 ) +
            (probablities.count_im_absoluted_immediately * self.w_im_absoluted_1 ) +
            (probablities.count_i_punish_after2 * self.w_i_punish_2 ) +
            (probablities.count_im_punished_after2 * self.w_im_punished_2 )
    }

    pub fn new_rand<R: Rng + ?Sized>(rng: &mut R) -> Self{
        rand::distr::StandardUniform{}.sample(rng)
    }

    pub(crate)fn _cross(&self, other: &Self) -> Self{
        Self{
            w_up_v_up: (self.w_up_v_up + other.w_up_v_up)/2.0,
            w_down_v_up: (self.w_down_v_up + other.w_down_v_up)/2.0,
            w_up_v_down: (self.w_up_v_down + other.w_up_v_down)/2.0,
            w_down_v_down: (self.w_down_v_down + other.w_down_v_down)/2.0,
            w_i_punish_1: (self.w_i_punish_1 + other.w_i_punish_1)/2.0,
            w_im_punished_1: (self.w_im_punished_1 + other.w_im_punished_1)/2.0,
            w_i_absolute_1: (self.w_i_absolute_1 + other.w_i_absolute_1)/2.0,
            w_im_absoluted_1: (self.w_im_absoluted_1 + other.w_im_absoluted_1)/2.0,
            w_i_punish_2: (self.w_i_punish_2 + other.w_i_punish_2)/2.0,
            w_im_punished_2: (self.w_im_punished_2 + other.w_im_punished_2)/2.0,
            stdev: (self.stdev + other.stdev)/2.0,
        }
    }
    pub(crate) fn _mutate_with_attribute_proba<ID: UsizeAgentId>(&mut self, probability: f64) -> Result<(), AmfiteatrError<ClassicScheme<ID>>>{
        let d = Bernoulli::new(probability)
            .map_err(|e| AmfiteatrError::Custom(e.to_string()))?;
        let mut rng = rand::rng();
        //let mut r = Self::default();
        if d.sample(&mut rng) {
            self.w_up_v_up = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_down_v_up = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_up_v_down = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_down_v_down = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_i_punish_1 = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_im_punished_1 = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_i_absolute_1 = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_im_absoluted_1 = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_i_punish_2 = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.w_im_punished_2 = rng.random_range(-1.0 .. 1.0);
        }
        if d.sample(&mut rng) {
            self.stdev = rng.random_range(0.0 .. 11.0);
        }

        Ok(())
    }
}

impl Default for GpClassic{
    fn default() -> Self {
        Self{
            w_up_v_up: 0.0,
            w_down_v_up: 0.0,
            w_up_v_down: 0.0,
            w_down_v_down: 0.0,
            w_i_punish_1: 0.0,
            w_im_punished_1: 0.0,
            w_i_absolute_1: 0.0,
            w_im_absoluted_1: 0.0,
            w_i_punish_2: 0.0,
            w_im_punished_2: 0.0,
            stdev: 1.0,
        }
    }
}
impl Distribution<GpClassic> for rand::distr::StandardUniform{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> GpClassic {

        //let w_uo_v_up = rng.random_range((-1.0, 1.0));

        GpClassic{
            w_up_v_up: rng.random_range(-1.0 .. 1.0),
            w_down_v_up: rng.random_range(-1.0 .. 1.0),
            w_up_v_down: rng.random_range(-1.0 .. 1.0),
            w_down_v_down: rng.random_range(-1.0 .. 1.0),
            w_i_punish_1: rng.random_range(-1.0 .. 1.0),
            w_im_punished_1: rng.random_range(-1.0 .. 1.0),
            w_i_absolute_1: rng.random_range(-1.0 .. 1.0),
            w_im_absoluted_1: rng.random_range(-1.0 .. 1.0),
            w_i_punish_2: rng.random_range(-1.0 .. 1.0),
            w_im_punished_2: rng.random_range(-1.0 .. 1.0),
            stdev: rng.random_range(0.0 .. 11.0),
        }
    }
}

impl<ID: UsizeAgentId> Policy< ClassicScheme<ID>> for GpClassic{
    type InfoSetType = LocalHistoryInfoSet<ID>;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<ClassicAction, AmfiteatrError<ClassicScheme<ID>>> {

        let gauss_expected = self._policy_get_gauss_mean(state);
        let sigma = self.stdev;

        let distr = rand_distr::Normal::new(gauss_expected, sigma).map_err(|e|
            AmfiteatrError::Custom(e.to_string()))?;



        let v = distr.sample(&mut rand::rng());

        if v > 0.0 {
            Ok(ClassicAction::Up)
        } else {
            Ok(ClassicAction::Down)
        }
    }
}

impl<ID: UsizeAgentId> PolicySpecimen<ClassicScheme<ID>, ()> for GpClassic{

    fn cross(&self, other: &Self) -> Self {
        self._cross(other)
    }

    fn mutate(&mut self, _mutagen: ()) -> Result<(), AmfiteatrError<ClassicScheme<ID>>> {
        self._mutate_with_attribute_proba(0.2)


    }
}
