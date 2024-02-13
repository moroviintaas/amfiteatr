use std::collections::HashMap;
use std::sync::Arc;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use amfiteatr_core::domain::{Renew};
use amfiteatr_core::env::{EnvironmentStateUniScore, EnvironmentStateSequential};
use log::{debug, trace};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use serde::Serialize;
use amfiteatr_core::error::AmfiError;
use crate::domain::{AgentNum, ClassicAction, ClassicGameDomain, ClassicGameError, ClassicGameUpdate, EncounterReport, IntReward, UsizeAgentId};
use crate::domain::ClassicGameError::ActionAfterGameOver;
use crate::{AsymmetricRewardTableInt, Side};



/// Structure to make note of player pairing - has information of other player, performed actions
/// (by this player) and [`Side`] on which player was paired.
#[derive(Copy, Clone, Debug, Default, Serialize, speedy::Writable, speedy::Readable)]
pub struct PlayerPairing<ID: UsizeAgentId> {
    pub paired_player: ID,
    pub taken_action: Option<ClassicAction>,
    pub side: Side
}

impl<ID: UsizeAgentId> Display for PlayerPairing<ID>{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        //write!(f, "({}-{})", self.id)
        let a = match self.taken_action{
            None => String::from("0"),
            Some(a) => format!("{:?}", a)
        };
        match self.side{
            Side::Left => write!(f, "[{} -> {}]", &a, self.paired_player),
            Side::Right => write!(f, "[{} <- {}]", self.paired_player, &a),

        }
    }
}
/// Alias for `Vec<PlayerPairing<ID>>`
pub type PairingVec<ID> = Vec<PlayerPairing<ID>>;

/// This is state of game prepared for many players and many rounds.
/// > It follows:
/// 1. It demands that the number of players is even.
/// 2. For every round shuffles players and match them in pairs. Players are informed with whom they are paired.
/// 3. Every pair makes new encounter.
/// 4. Every player is subsequently asked to make action which is noted.
/// 5. After all players moved, reports of every encounter is prepared and sent to all players.
/// (Every player get complete information about all encounters, what he does with this knowledge
/// is up to his information set implementation).
///
#[derive(Debug, Clone, Serialize)]
pub struct PairingState<ID: UsizeAgentId>{
    actual_pairings: PairingVec<ID>,
    previous_pairings: Vec<Arc<PairingVec<ID>>>,
    target_rounds: usize,
    indexes: Vec<usize>,
    reward_table: AsymmetricRewardTableInt,
    score_cache: Vec<i64>,
    current_player_index: usize,
    _id: PhantomData<ID>


}
/// Alias for `PairingState<AgentNum>`
pub type PairingStateNumbered = PairingState<AgentNum>;

impl<ID: UsizeAgentId> PairingState<ID>{
    pub fn new_even(players: usize, target_rounds: usize, reward_table: AsymmetricRewardTableInt) -> Result<Self, ClassicGameError<ID>>{
        /*
        if players & 0x01 != 0{
            return Err(ClassicGameError::ExpectedEvenNumberOfPlayers(players));
        }


         */

        let mut indexes: Vec<usize> = (0..players).into_iter().collect();
        let mut rng = thread_rng();
        indexes.shuffle(&mut rng);
        //debug!("Shuffled indexes: {:?}", &indexes);
        //println!("Shuffled indexes: {:?}", &indexes);
        let actual_pairings = Self::create_pairings(&indexes[..])?;

        let mut score_cache = Vec::with_capacity(indexes.len());
        score_cache.resize_with(indexes.len(), || 0);
        Ok(Self{
            actual_pairings,
            indexes,
            target_rounds,
            previous_pairings: Vec::with_capacity(target_rounds),
            reward_table,
            score_cache,
            current_player_index: 0,
            _id: PhantomData::default()
        })
    }

    fn create_pairings(indexes: &[usize]) -> Result<PairingVec<ID>, ClassicGameError<ID>>{
        if indexes.len() & 0x01 != 0{
            return Err(ClassicGameError::ExpectedEvenNumberOfPlayers(indexes.len() as u32));
        } else {
            let mut v = Vec::with_capacity(indexes.len());
            v.resize_with(indexes.len(), || PlayerPairing{
                paired_player: ID::make_from_usize(0),
                taken_action: None,
                side: Default::default(),
            }) ;
            for i in 0..indexes.len(){
                let index:usize = indexes[i] as usize;
                if i & 0x01 == 0{


                    //even
                    v[index] = PlayerPairing{
                        paired_player: ID::make_from_usize(indexes[i+1] as usize),
                        taken_action: None,
                        side: Side::Left,
                    }

                } else {

                    v[index] = PlayerPairing{
                        paired_player: ID::make_from_usize(indexes[i-1] as usize),
                        taken_action: None,
                        side: Side::Right,
                    }
                }
            }
            Ok(v)
        }

    }

    fn prepare_new_pairing(&mut self) -> Result<(), ClassicGameError<ID>>{

        let mut rng = thread_rng();
        self.indexes.shuffle(&mut rng);
        debug!("Preparing new pairings for indexes: {:?}", self.indexes);
        //debug!("Shuffled indexes: {:?}", &self.indexes);
        //println!("Shuffled indexes: {:?}", &self.indexes);
        let mut pairings = Self::create_pairings(&self.indexes[..])?;
        std::mem::swap(&mut pairings, &mut self.actual_pairings);
        //debug!("Pairings: {:?}", &self.actual_pairings);
        //println!("Pairings: {:?}", &self.actual_pairings);
        self.previous_pairings.push(Arc::new(pairings));


        Ok(())

    }

    pub fn is_round_clean(&self) -> bool{
        self.current_player_index == 0
    }

}

impl<ID: UsizeAgentId> Display for PairingState<ID>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        /*write!(f, "Rounds: {} |", self.previous_pairings.len())?;
        let mut s = self.previous_pairings.iter().fold(String::new(), |mut acc, update| {
            acc.push_str(&format!("({}){:#}-{:#}\n", update.side, update.own_action, update.other_player_action));
            acc
        });
        s.pop();
        write!(f, "{}", s)*/

        for r in 0..self.previous_pairings.len(){
            write!(f, "Round: {r:}:\n")?;
            for i in 0..self.previous_pairings[r].len(){
                write!(f, "\t{}\tpositioned: {:?}\tpaired with: {}\t;",
                       i, self.previous_pairings[r][i].side, self.previous_pairings[r][i].paired_player)?;
                if let Some(action) = self.previous_pairings[r][i].taken_action{
                    write!(f, "taken action: {action:?}\t")?;
                }
                else{
                    write!(f, "taken action: ---\t")?;
                }
                let other_index = self.previous_pairings[r][i].paired_player.as_usize();
                if let Some(action) = self.previous_pairings[r][other_index].taken_action{
                    write!(f, "against: {action:?}\t")?;
                }
                else{
                    write!(f, "against: ---\t")?;
                }
                write!(f, "\n")?;
            }
        }
        write!(f, "")
    }
}

impl<ID: UsizeAgentId> EnvironmentStateSequential<ClassicGameDomain<ID>> for PairingState<ID> {
    type Updates = Vec<(ID, ClassicGameUpdate<ID>)>;

    fn current_player(&self) -> Option<ID> {
        if self.is_finished(){
            return None;
        }
        if self.current_player_index  < self.actual_pairings.len(){
            Some(ID::make_from_usize(self.current_player_index))
        } else {
            None
        }
    }

    fn is_finished(&self) -> bool {
        self.previous_pairings.len() >= self.target_rounds
    }

    fn forward(&mut self, agent: ID, action: ClassicAction)
        -> Result<Self::Updates, ClassicGameError<ID>> {
        if let Some(destined_agent) = self.current_player(){
            if destined_agent == agent{
                debug!("Forwarding environment with agent {agent:} action: {action:?}, ");
                self.actual_pairings[agent.as_usize()].taken_action = Some(action);
                let this_pairing = self.actual_pairings[agent.as_usize()];
                let other_player_index = this_pairing.paired_player;
                let other_pairing = self.actual_pairings[other_player_index.as_usize()];
                // possibly update score cache if other player played already
                if let Some(other_action) = other_pairing.taken_action {
                    let (left_action, right_action) = match this_pairing.side{
                        Side::Left => (action, other_action),
                        Side::Right => (other_action, action)
                    };
                    let rewards = self.reward_table.rewards(left_action, right_action);
                    let rewards_reoriented = match this_pairing.side{
                        Side::Left => rewards,
                        Side::Right => (rewards.1, rewards.0)
                    };
                    self.score_cache[agent.as_usize()] += rewards_reoriented.0;
                    self.score_cache[other_player_index.as_usize()] += rewards_reoriented.1;


                }
                //set next index
                self.current_player_index +=1;
                debug!("Next player index would be {:?}", self.current_player_index);
                if self.current_player_index >= self.actual_pairings.len(){



                    let encounters_vec: HashMap<ID, EncounterReport<ID>> = (0..self.actual_pairings.len())
                        .into_iter().map(|i|{
                        let actual_pairing = self.actual_pairings[i];
                        let other_player = self.actual_pairings[i].paired_player;
                        //let reverse_pairing = self.actual_pairings[other_player.as_usize()];
                        (ID::make_from_usize(i), EncounterReport{
                            own_action: self.actual_pairings[i].taken_action.unwrap(),
                            other_player_action: self.actual_pairings[other_player.as_usize()].taken_action.unwrap(),
                            side: actual_pairing.side,
                            other_id: other_player,
                        })
                    }).collect();
                    let encounters = Arc::new(encounters_vec);

                    self.prepare_new_pairing()?;
                    self.current_player_index = 0;
                    trace!("Played rounds so far: {}", self.previous_pairings.len());
                    debug!("Last player in round played, preparing new round, setting player index to 0");

                    let opairings = match self.is_finished(){
                        true => None,
                        false => Some(Arc::new(self.actual_pairings.clone()))
                    };
                    let singe_update = ClassicGameUpdate{
                        encounters,
                        pairing: opairings,
                    };
                    let updates: Vec<(ID, ClassicGameUpdate<ID>)> = (0..self.actual_pairings.len())
                        .into_iter().map(|i|{
                        (ID::make_from_usize(i), singe_update.clone())
                    }).collect();

                    trace!("Finishing round. Now after: {}", self.previous_pairings.len());
                    Ok(updates)

                } else{
                    Ok(Vec::default())
                }





            } else{
                Err(ClassicGameError::GameViolatedOrder { acted: agent, expected: self.current_player() })
            }

        } else {
            Err(ActionAfterGameOver(agent))
        }

    }
}

impl<ID: UsizeAgentId> EnvironmentStateUniScore<ClassicGameDomain<ID>> for PairingState<ID> {
    fn state_score_of_player(&self, agent: &ID) -> IntReward {
        self.score_cache[agent.as_usize()]
    }
}

impl<ID: UsizeAgentId> Renew<ClassicGameDomain<ID>, ()> for PairingState<ID>{
    fn renew_from(&mut self, _base: ())  -> Result<(), AmfiError<ClassicGameDomain<ID>>> {
        debug!("Renewing state");
        //self.score_cache.iter_mut().for_each(|a|*a=0);
        for i in 0..self.score_cache.len(){
            self.score_cache[i] = 0;
        }
        self.previous_pairings.clear();
        self.current_player_index = 0;
        let mut rng = thread_rng();
        self.indexes.shuffle(&mut rng);
        self.actual_pairings = Self::create_pairings(&self.indexes[..]).unwrap();
        debug!("After renewing state, with pairings of length = {}", self.actual_pairings.len());
        Ok(())
    }
}