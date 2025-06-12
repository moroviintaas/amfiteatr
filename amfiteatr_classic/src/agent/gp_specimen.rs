//use crate::agent::LocalHistoryInfoSet;
//use crate::domain::UsizeAgentId;

#[derive(Copy, Clone, Debug)]
pub struct EventProbabilities{
    pub p_up_v_up: f64,
    pub p_up_v_down: f64,
    pub proba_down_v_up: f64,
    pub proba_down_v_down: f64,

    pub proba_im_punished_immediately: f64,
    pub proba_i_punish_immediately: f64,

    pub proba_im_punished_after2: f64,
    pub proba_i_punish_after2: f64,

    pub proba_im_absoluted_immediately: f64,
    pub proba_i_absolute_immediately: f64,




}

pub struct GpClassic{

}

impl GpClassic{

    /*
    fn event_probabilities<ID: UsizeAgentId>(&self, info_set: &LocalHistoryInfoSet<ID>){

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
    
     */

}