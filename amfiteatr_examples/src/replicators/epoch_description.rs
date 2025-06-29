use std::collections::HashMap;
use amfiteatr_classic::domain::AgentNum;
use amfiteatr_rl::policy::LearnSummary;


#[derive(Default, Debug, Clone)]
pub struct EpochDescription {
    pub(crate) scores: HashMap<AgentNum, Vec<i64>>,

    pub(crate)  network_learning_hawk_moves: HashMap<AgentNum, Vec<usize>>,
    pub(crate)  network_learning_dove_moves: HashMap<AgentNum, Vec<usize>>,

}



pub fn hash_map_average_i64(hm: &HashMap<AgentNum, Vec<i64>>, round_divider: Option<usize>) -> HashMap<AgentNum, f64>{
    hm.iter().filter_map(|(a, vec)| {
            match vec.is_empty(){
                true => None,
                false => Some((a, vec))
            }
        }).map(|(a, vec)| (*a, vec.iter().sum::<i64>() as f64 / vec.len() as f64 / round_divider.unwrap_or(1) as f64))
        .collect()
}

pub fn hash_map_average_usize(hm: &HashMap<AgentNum, Vec<usize>>, round_divider: Option<usize>) -> HashMap<AgentNum, f64>{
    hm.iter().filter_map(|(a, vec)| {
        match vec.is_empty(){
            true => None,
            false => Some((a, vec))
        }
    }).map(|(a, vec)| (*a, vec.iter().sum::<usize>() as f64 / vec.len() as f64/ round_divider.unwrap_or(1) as f64))
        .collect()
}
impl EpochDescription {

    pub fn mean(&self) -> EpochDescriptionMean{

        let mean_scores: HashMap<AgentNum, f64> = hash_map_average_i64(&self.scores, None);
        let mean_network_learning_hawk_moves = hash_map_average_usize(&self.network_learning_hawk_moves, None);
        let mean_network_learning_dove_moves = hash_map_average_usize(&self.network_learning_dove_moves, None);

        EpochDescriptionMean{
            mean_scores,
            mean_network_learning_hawk_moves,
            mean_network_learning_dove_moves
        }
    }
    pub fn mean_divide_round(&self, rounds: usize) -> EpochDescriptionMean{
        let mean_scores: HashMap<AgentNum, f64> = hash_map_average_i64(&self.scores, Some(rounds));
        let mean_network_learning_hawk_moves = hash_map_average_usize(&self.network_learning_hawk_moves, Some(rounds));
        let mean_network_learning_dove_moves = hash_map_average_usize(&self.network_learning_dove_moves, Some(rounds));

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

pub type SessionDescription = Vec<EpochDescriptionMean>;
pub type SessionLearningSummaries = Vec<LearnSummary>;