use log::warn;
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::env::{EnvironmentStateSequential, EnvironmentStateUniScore};
use amfiteatr_core::error::AmfiteatrError;
use crate::common::{Board, ConnectFourAction, ConnectFourBinaryObservation, ConnectFourDomain, ConnectFourError, ConnectFourPlayer};


#[derive(Clone, Debug)]
pub struct ConnectFourRustEnvState{
    board: Board,

    current_player: Option<ConnectFourPlayer>,
    truncations: [bool;2],
    terminations: [bool;2],
    scores: [f32;2],
    step: u32,
    winner: [bool;2],
    render: bool

}

impl Default for ConnectFourRustEnvState{
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectFourRustEnvState{
    pub fn new() -> Self{
        Self{
            board: Board::default(),
            current_player: Some(ConnectFourPlayer::One),
            truncations: [false, false],
            terminations: [false, false],
            scores: [0.0, 0.0],
            step: 0,
            winner: [false, false],
            render: false
        }
    }
    pub fn steps_completed(&self) -> u32{
        self.step
    }

    pub fn winner(&self) -> Option<ConnectFourPlayer>{
        if self.winner[0] {
            return Some(ConnectFourPlayer::One);
        }
        if self.winner[1] {
            return Some(ConnectFourPlayer::Two);
        }
        return None

    }

    #[inline]
    fn is_board_full(&self) -> bool{
        for i in 0..self.board[0].len(){
            if self.board[0][i].is_none(){
                return false
            }
        }
        return true
    }
    #[inline]
    fn check_for_winner(&self, player: ConnectFourPlayer) -> bool{
        // check horizontal
        let column_count = 7;
        let row_count = 6;
        for c in 0..column_count-3{
            for r in 0..row_count{
                if self.board[r][c] == Some(player) &&
                    self.board[r][c+1] == Some(player) &&
                    self.board[r][c+2] == Some(player) &&
                    self.board[r][c+3] == Some(player){

                    return true;
                }
            }
        }

        // check vertical
        for c in 0..column_count{
            for r in 0..row_count - 3{
                if self.board[r][c] == Some(player) &&
                    self.board[r+1][c] == Some(player) &&
                    self.board[r+2][c] == Some(player) &&
                    self.board[r+3][c] == Some(player){

                    return true;
                }
            }
        }

        // check positively sloped

        for c in 0..column_count-3{
            for r in 0..row_count - 3{
                if self.board[r][c] == Some(player) &&
                    self.board[r+1][c+1] == Some(player) &&
                    self.board[r+2][c+1] == Some(player) &&
                    self.board[r+3][c+1] == Some(player){

                    return true;
                }
            }
        }
        // check negatively sloped
        for c in 0..column_count-3{
            for r in 3..row_count{
                if self.board[r][c] == Some(player) &&
                    self.board[r-1][c+1] == Some(player) &&
                    self.board[r-2][c+1] == Some(player) &&
                    self.board[r-3][c+1] == Some(player){

                    return true;
                }
            }
        }



        return false
    }
}

impl EnvironmentStateSequential<ConnectFourDomain> for ConnectFourRustEnvState{
    type Updates = [(<ConnectFourDomain as DomainParameters>::AgentId, <ConnectFourDomain as DomainParameters>::UpdateType );1];

    fn current_player(&self) -> Option<ConnectFourPlayer> {
        self.current_player
    }

    fn is_finished(&self) -> bool {
        self.current_player.is_none()
    }

    fn forward(&mut self, agent: ConnectFourPlayer, action: ConnectFourAction) -> Result<Self::Updates, ConnectFourError> {

        let column = action.index();
        if Some(agent) != self.current_player{
            return Err(ConnectFourError::PlayerViolatedOrder {player: agent})
        }

        if self.board[0][action.index()].is_some(){
            return Err(ConnectFourError::IllegalActionFullColumn{
                player: agent,
                action
            })
        }

        for row in (0..6).rev(){
            if self.board[row][column].is_none(){
                self.board[row][column] = Some(agent);
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

        #[no_mangle]
        if self.render{
            warn!("Rendering is not suppoerted, this is operation placeholder")
        }

        Ok([(agent.other(), ConnectFourBinaryObservation::build_from(&self.board, agent.other()))])

    }
}

impl Renew<ConnectFourDomain, ()> for ConnectFourRustEnvState{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ConnectFourDomain>> {
        self.board = Board::default();
        self.current_player = Some(ConnectFourPlayer::One);
        self.truncations = [false, false];
        self.terminations = [false, false];
        self.scores = [0.0, 0.0];
        self.step = 0;
        self.winner = [false,false];
        Ok(())
    }
}

impl EnvironmentStateUniScore<ConnectFourDomain> for ConnectFourRustEnvState{
    fn state_score_of_player(&self, agent: &<ConnectFourDomain as DomainParameters>::AgentId)
        -> <ConnectFourDomain as DomainParameters>::UniversalReward {
        self.scores[agent.index()]
    }
}