use std::collections::HashMap;
use amfiteatr_classic::domain::AgentNum;




#[derive(Default, Debug, Clone)]
pub struct EpochDescription {
    pub(crate) scores: HashMap<AgentNum, Vec<i64>>,

    pub(crate)  network_learning_hawk_moves: HashMap<AgentNum, Vec<usize>>,
    pub(crate)  network_learning_dove_moves: HashMap<AgentNum, Vec<usize>>,

}


pub fn hash_map_average_i64(hm: &HashMap<AgentNum, Vec<i64>>) -> HashMap<AgentNum, f64>{
    hm.iter().filter_map(|(a, vec)| {
            match vec.is_empty(){
                true => None,
                false => Some((a, vec))
            }
        }).map(|(a, vec)| (*a, vec.iter().sum::<i64>() as f64 / vec.len() as f64))
        .collect()
}

pub fn hash_map_average_usize(hm: &HashMap<AgentNum, Vec<usize>>) -> HashMap<AgentNum, f64>{
    hm.iter().filter_map(|(a, vec)| {
        match vec.is_empty(){
            true => None,
            false => Some((a, vec))
        }
    }).map(|(a, vec)| (*a, vec.iter().sum::<usize>() as f64 / vec.len() as f64))
        .collect()
}
impl EpochDescription {

    pub fn mean(&self) -> EpochDescriptionMean{

        let mean_scores: HashMap<AgentNum, f64> = hash_map_average_i64(&self.scores);
        let mean_network_learning_hawk_moves = hash_map_average_usize(&self.network_learning_hawk_moves);
        let mean_network_learning_dove_moves = hash_map_average_usize(&self.network_learning_dove_moves);

        EpochDescriptionMean{
            mean_scores,
            mean_network_learning_hawk_moves,
            mean_network_learning_dove_moves
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct EpochDescriptionMean{
    pub(crate) mean_scores: HashMap<AgentNum, f64>,

    pub(crate)  mean_network_learning_hawk_moves: HashMap<AgentNum, f64>,
    pub(crate)  mean_network_learning_dove_moves: HashMap<AgentNum, f64>,
}
