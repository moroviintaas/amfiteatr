use log::warn;
use ndarray::{Array2, Array3};
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::env::{GameStateWithPayoffs, SequentialGameState};
use amfiteatr_core::error::AmfiteatrError;
use crate::connect_four::common::{Board, ConnectFourAction, ConnectFourBinaryObservation, ConnectFourDomain, ConnectFourError, ConnectFourPlayer};
use crate::connect_four::env::ConnectFourRustEnvState;

#[derive(Clone, Debug)]
pub struct ConnectFourRustNdEnvState{
    board: Array2<u8>,

    current_player: Option<ConnectFourPlayer>,
    truncations: [bool;2],
    terminations: [bool;2],
    scores: [f32;2],
    step: u32,
    winner: [bool;2],
    render: bool

}


impl Default for ConnectFourRustNdEnvState{
    fn default() -> Self {
        Self::new()
    }
}
impl ConnectFourRustNdEnvState{
    pub fn new() -> Self{
        Self{
            board: Array2::zeros((6,7)),
            current_player: Some(ConnectFourPlayer::One),
            truncations: [false, false],
            terminations: [false, false],
            scores: [0.0, 0.0],
            step: 0,
            winner: [false, false],
            render: false
        }
    }

    #[allow(dead_code)]
    pub fn steps_completed(&self) -> u32{
        self.step
    }

    #[allow(dead_code)]
    pub fn winner(&self) -> Option<ConnectFourPlayer>{
        if self.winner[0] {
            return Some(ConnectFourPlayer::One);
        }
        if self.winner[1] {
            return Some(ConnectFourPlayer::Two);
        }
        None

    }

    #[inline]
    fn is_board_full(&self) -> bool{
        !self.board.iter().any(|i| i == &0)

    }
    #[inline]
    fn check_for_winner(&self, player: ConnectFourPlayer) -> bool{

        let player_id = player.index() as u8 + 1 ;
        // check horizontal
        let column_count = 7;
        let row_count = 6;
        for c in 0..column_count-3{
            for r in 0..row_count{
                if self.board[(r, c)] == player_id &&
                    self.board[(r, c+1)] == player_id &&
                    self.board[(r, c+2)] == player_id &&
                    self.board[(r, c+3)] == player_id{

                    return true;
                }
            }
        }

        // check vertical
        for c in 0..column_count{
            for r in 0..row_count - 3{
                if self.board[(r, c)]== player_id&&
                    self.board[(r+1, c)] == player_id&&
                    self.board[(r+2, c)] == player_id &&
                    self.board[(r+3, c)] == player_id{

                    return true;
                }
            }
        }

        // check positively sloped

        for c in 0..column_count-3{
            for r in 0..row_count - 3{
                if self.board[(r, c)] == player_id&&
                    self.board[(r+1, c+1)] == player_id &&
                    self.board[(r+2, c+1)] == player_id &&
                    self.board[(r+3, c+1)] == player_id{

                    return true;
                }
            }
        }
        // check negatively sloped
        for c in 0..column_count-3{
            for r in 3..row_count{
                if self.board[(r,c)] == player_id &&
                    self.board[(r-1,c+1)]  == player_id&&
                    self.board[(r-2,c+1)] == player_id &&
                    self.board[(r-3,c+1)] == player_id{

                    return true;
                }
            }
        }



        false
    }
}

impl SequentialGameState<ConnectFourDomain> for ConnectFourRustNdEnvState{
    type Updates = [(<ConnectFourDomain as DomainParameters>::AgentId, <ConnectFourDomain as DomainParameters>::UpdateType );1];

    fn current_player(&self) -> Option<ConnectFourPlayer> {
        self.current_player
    }

    fn is_finished(&self) -> bool {
        self.current_player.is_none()
    }

    fn forward(&mut self, agent: ConnectFourPlayer, action: ConnectFourAction) -> Result<Self::Updates, ConnectFourError> {

        let agent_piece = agent.index() as u8 + 1;
        let column = action.index();
        if Some(agent) != self.current_player{
            return Err(ConnectFourError::PlayerViolatedOrder {player: agent})
        }

        if self.board[(0, action.index())] != 0{
            return Err(ConnectFourError::IllegalActionFullColumn{
                player: agent,
                action
            })
        }

        for row in (0..6).rev(){
            if self.board[(row,column)] == 0{
                self.board[(row,column)] = agent_piece;
                break;
            }
        }

        let winner = self.check_for_winner(agent);
        if winner{
            self.scores[agent.index()] += 1.0;
            self.scores[agent.other().index()] -= 1.0;
            self.terminations = [true, true];
            self.current_player = None;
            self.winner[agent.index()] = true;
        } else if self.is_board_full(){
            self.terminations = [true, true];
            self.current_player = None;
        } else {
            self.current_player = Some(agent.other());
        }


        if self.render{
            //#[no_mangle]
            warn!("Rendering is not supported, this is operation placeholder")
        }

        Ok([(agent.other(), ConnectFourBinaryObservation::build_from_nd(&self.board, agent.other()))])

    }


}

impl Renew<ConnectFourDomain, ()> for ConnectFourRustNdEnvState{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ConnectFourDomain>> {
        self.board = Array2::zeros((6,7));
        self.current_player = Some(ConnectFourPlayer::One);
        self.truncations = [false, false];
        self.terminations = [false, false];
        self.scores = [0.0, 0.0];
        self.step = 0;
        self.winner = [false,false];
        Ok(())
    }
}

impl GameStateWithPayoffs<ConnectFourDomain> for ConnectFourRustNdEnvState{
    fn state_payoff_of_player(&self, agent: &<ConnectFourDomain as DomainParameters>::AgentId)
                              -> <ConnectFourDomain as DomainParameters>::UniversalReward {
        self.scores[agent.index()]
    }
}