use std::collections::HashMap;
use crate::scheme::{Reward, Scheme};
use getset::{CopyGetters, Getters, Setters};
#[derive(Getters, CopyGetters, Setters)]
pub struct GameSummaryGen<S: Scheme>{
    #[getset(get = "pub")]
    scores: HashMap<S::AgentId, S::UniversalReward>,
    #[getset(get_copy = "pub")]
    steps: u64,
    #[getset(get = "pub", set = "pub")]
    violating_agent: Option<S::AgentId>,
}

impl<S: Scheme> GameSummaryGen<S>{
    pub fn new(scores: HashMap<S::AgentId, S::UniversalReward>, steps: u64, violating_agent: Option<S::AgentId>) -> GameSummaryGen<S>{
        GameSummaryGen{scores, steps, violating_agent}
    }
}



#[derive(Getters)]
pub struct EpochSummaryGen<S: Scheme> {
    //score_sums: HashMap<S::AgentId, S::UniversalReward>,
    //steps: Vec<f64>
    #[getset(get = "pub")]
    game_summaries: Vec<GameSummaryGen<S>>,
    #[getset(get = "pub")]
    //faults: Vec<Option<S::AgentId>>
    faults: HashMap<S::AgentId, u64>
}

impl<S: Scheme> EpochSummaryGen<S>{

    pub fn new(game_summaries: Vec<GameSummaryGen<S>>) -> EpochSummaryGen<S>{

        let mut faults = HashMap::new();

        for s in &game_summaries{
            if let Some(agent) = s.violating_agent(){
                let previous_faults = faults.entry(agent.clone()).or_insert(0);
                *previous_faults += 1;
            }
        }

        Self{
            game_summaries,
            faults
            //faults: HashMap::new()
        }

    }

    /*
    pub fn new_with_ids_of_invalid_actions(game_summaries_and_faulers: Vec<(GameSummaryGen<S>, >) -> Self{

        let mut faults = HashMap::new();
        for (_, oid) in &game_summaries_and_faulers{
            if let Some(id) = oid{
                let mut previous_faults = faults.entry(id.clone()).or_insert(0);
                *previous_faults += 1;
            }
        }
        let game_summaries = game_summaries_and_faulers.into_iter().map(|(s, _)|{
            s
        }).collect();

        Self{
            game_summaries, faults
        }
    }

     */

    pub fn sum_score(&self, agent_id: &<S as Scheme>::AgentId) -> Option<<S as Scheme>::UniversalReward>{
        let mut participated = false;
        let sum = self.game_summaries.iter()
            .fold(<<S as Scheme>::UniversalReward as Reward>::neutral(), |acc, x|{
                x.scores().get(agent_id)
                    .map_or_else(
                        <<S as Scheme>::UniversalReward as Reward>::neutral,
                        |s| { participated = true; acc+s}
                    )
            });

        if participated{
            Some(sum)
        }
        else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.game_summaries.len()
    }

    pub fn is_empty(&self) -> bool{
        self.game_summaries.is_empty()
    }
}
