use std::ops::Div;
use rand::distr::{Distribution, Open01};
use rand::Rng;
use amfiteatr_core::agent::Policy;
use amfiteatr_core::error::AmfiteatrError;
use crate::agent::LocalHistoryInfoSet;
use crate::domain::{ClassicAction, ClassicGameDomain, UsizeAgentId};

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

impl<ID: UsizeAgentId> Policy< ClassicGameDomain<ID>> for GpClassic{
    type InfoSetType = LocalHistoryInfoSet<ID>;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<ClassicAction, AmfiteatrError<ClassicGameDomain<ID>>> {

        let gauss_expected = self._policy_get_gauss_mean(state);
        let sigma = self.stdev;

        todo!()
    }
}

impl GpClassic{



    


}