use std::ops::Div;
use crate::agent::LocalHistoryInfoSet;
use crate::domain::UsizeAgentId;

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

}

impl GpClassic{


    fn event_countbilities<ID: UsizeAgentId>(&self, info_set: &LocalHistoryInfoSet<ID>){

        let mut uu = 0f64;
        let mut ud = 0f64;
        let mut du = 0f64;
        let mut dd = 0f64;
        let mut imp = 0f64;
        let mut ip = 0f64;
        let mut imp2 = 0f64;
        let mut ip2 = 0f64;
        let mut ia = 0f64;
        let ima = 0f64;



    }
    


}