use std::fmt::{Display, Formatter};
use amfiteatr_core::agent::AgentIdentifier;
use amfiteatr_core::domain::{Action, DomainParameters};
use amfiteatr_core::error::{AmfiteatrError, ConvertError};
use amfiteatr_rl::error::{AmfiteatrRlError, TensorRepresentationError};
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::TryIntoTensor;

pub type ErrorRL = AmfiteatrRlError<ConnectFourDomain>;
pub type ErrorAmfi = AmfiteatrError<ConnectFourDomain>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConnectFourPlayer{
    One,
    Two
}

impl Display for ConnectFourPlayer{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}",  match self{ Self::One => "one", Self::Two => "two"})
    }
}

impl AgentIdentifier for ConnectFourPlayer{

}

impl ConnectFourPlayer{
    pub fn index(&self) -> usize{
        match self{
            ConnectFourPlayer::One => 0,
            ConnectFourPlayer::Two => 1,
        }
    }

    pub fn other(&self) -> Self{
        match self{
            ConnectFourPlayer::One => Self::Two,
            ConnectFourPlayer::Two => Self::One
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectFourDomain{}

#[derive(Debug, Copy, Clone)]
pub struct ConnectFourAction{
    id: u8
}


impl TryFrom<i64> for ConnectFourAction{
    type Error = ConvertError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        match value{
            a @ 0..=6 => Ok(Self{id: a as u8}),
            different => Err(ConvertError::ActionDeserialize(format!("Failed converting number {different:} to ConnectFourAction")))
        }
    }
}
impl ConnectFourAction{
    pub fn index(&self) -> usize{
        self.id as usize
    }
}

const ALL_ACTIONS: [ConnectFourAction;7] = [
    ConnectFourAction{id:0},
    ConnectFourAction{id:1},
    ConnectFourAction{id:2},
    ConnectFourAction{id:3},
    ConnectFourAction{id:4},
    ConnectFourAction{id:5},
    ConnectFourAction{id:6},
];
impl Display for ConnectFourAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column {:?}", self.id+1)
    }
}
impl Action for ConnectFourAction {}

impl TryIntoTensor for ConnectFourAction {
    fn try_to_tensor(&self) -> Result<Tensor, TensorRepresentationError> {
        Ok(Tensor::from_slice(&[self.id as f32]))
    }
}

impl TryFrom<&Tensor> for ConnectFourAction{
    type Error = ConvertError;

    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        let v: Vec<i64> = match Vec::try_from(value){
            Ok(v) => v,
            Err(e) => {
                return Err(ConvertError::ActionDeserialize(format!("{}: {e:}", value)))
            }
        };
        let action_index = v[0];
        ConnectFourAction::try_from(action_index)

    }
}
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConnectFourError {
    #[error("Game not initialized. Use reseed().")]
    GameStateNotInitialized,
    #[error("Player {player:} added to full column of index{action:?}")]
    IllegalActionFullColumn{
        player: ConnectFourPlayer,
        action: ConnectFourAction
    },
    #[error("Player {player:}'s dead step")]
    PlayerDeadStep{
        player: ConnectFourPlayer,
    },
    #[error("Internal game violated order by {player:}")]
    PlayerViolatedOrder{
        player: ConnectFourPlayer,
    },



}

impl ConnectFourError{
    pub fn fault_of_player(&self) -> Option<ConnectFourPlayer>{
        match self{
            Self::GameStateNotInitialized => None,
            Self::IllegalActionFullColumn { player, .. }
            | Self::PlayerDeadStep { player, .. }
            | Self::PlayerViolatedOrder{ player, ..} => Some(*player),
        }
    }
}

pub type BoardRow = [Option<ConnectFourPlayer>;7];
pub type Board = [BoardRow;6];


#[derive(Clone, Debug, Default)]
pub struct ConnectFourBinaryObservation{
    pub board: [[[u8;2];7];6]
}



impl ConnectFourBinaryObservation{

    pub fn build_from(board: &Board, for_agent: ConnectFourPlayer) -> Self{


        let mut b: [[[u8;2];7];6] = Default::default();
        for row in 0..board.len(){
            for column in 0..board[row].len(){
                b[row][column] = match board[row][column]{
                    None => [0,0],
                    Some(own) if  own == for_agent => [1,0],
                    Some(_other) => [0,1]
                };

            }
        }
        Self{board: b}



    }
}

impl DomainParameters for ConnectFourDomain{
    type ActionType = ConnectFourAction;
    type GameErrorType = ConnectFourError;
    type UpdateType = ConnectFourBinaryObservation;
    type AgentId = ConnectFourPlayer;
    type UniversalReward = f32;
}