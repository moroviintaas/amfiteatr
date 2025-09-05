use log::debug;
use amfiteatr_core::demo::DemoError;
use amfiteatr_core::scheme::{Scheme, Renew};
use amfiteatr_core::env::{GameStateWithPayoffs, SequentialGameState};
use amfiteatr_core::error::AmfiteatrError;
use crate::expensive_update::domain::{ExpensiveUpdateDomain, UpdateCost};


#[derive(Debug, Clone)]
pub struct ExpensiveUpdateState{
    current_player_index: u64,
    current_round: u64,
    max_number_of_rounds: u64,
    number_of_players: u64,

    small_update_cost_per_agent: UpdateCost,
    big_update_cost_per_agent: UpdateCost,
    big_update_cost_flat: UpdateCost,



}

impl ExpensiveUpdateState{
    pub fn new(number_of_rounds: u64, number_of_players: u64, small_update_cost_per_agent: UpdateCost, big_update_cost_per_agent: UpdateCost, big_update_cost_flat: UpdateCost) -> ExpensiveUpdateState{
        Self{
            current_player_index: 0,
            current_round: 0,
            max_number_of_rounds: number_of_rounds,
            number_of_players,
            small_update_cost_per_agent,
            big_update_cost_per_agent,
            big_update_cost_flat

        }
    }
}

impl SequentialGameState<ExpensiveUpdateDomain> for ExpensiveUpdateState{
    type Updates = Vec<(u64, UpdateCost)>;

    fn current_player(&self) -> Option<<ExpensiveUpdateDomain as Scheme>::AgentId> {
        if self.current_round < self.max_number_of_rounds{
            Some(self.current_player_index)
        } else {
            None
        }
    }

    fn is_finished(&self) -> bool {
        self.current_round >= self.max_number_of_rounds
    }

    fn forward(
        &mut self,
        agent: <ExpensiveUpdateDomain as Scheme>::AgentId,
        _action: <ExpensiveUpdateDomain as Scheme>::ActionType
    ) -> Result<Self::Updates, <ExpensiveUpdateDomain as Scheme>::GameErrorType> {

        if Some(agent) != self.current_player() || self.current_player() == None{
            return Err(DemoError(format!("Agent {agent} does not match current player {}", self.current_player_index)))
        }

        self.current_player_index += 1;
        if self.current_player_index == self.number_of_players{
            self.current_round+=1;
            self.current_player_index = 0;
            debug!("Ending round {}", self.current_round);
        }

        if self.small_update_cost_per_agent > 0{
            let mut updates = (0..self.number_of_players).map(|n| (n, self.small_update_cost_per_agent))
                .collect::<Vec<_>>();
            if self.big_update_cost_per_agent > 0{
                updates.push((self.current_player_index, (self.big_update_cost_per_agent*self.number_of_players) + self.big_update_cost_flat));
            }
            Ok(updates)
        } else {
            Ok(vec![(self.current_player_index, self.big_update_cost_per_agent*self.number_of_players+ self.big_update_cost_flat)])
        }
    }
}

impl GameStateWithPayoffs<ExpensiveUpdateDomain> for ExpensiveUpdateState{
    fn state_payoff_of_player(&self, _agent: &<ExpensiveUpdateDomain as Scheme>::AgentId) -><ExpensiveUpdateDomain as Scheme>::UniversalReward {
        0.0
    }
}

impl Renew<ExpensiveUpdateDomain, ()> for ExpensiveUpdateState{
    fn renew_from(&mut self, _base: ())  -> Result<(), AmfiteatrError<ExpensiveUpdateDomain>> {

        self.current_player_index = 0;
        self.current_round = 0;
        Ok(())
    }
}