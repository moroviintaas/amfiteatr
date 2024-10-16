use amfiteatr_core::agent::InformationSet;
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_rl::error::TensorRepresentationError;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::{ConversionToTensor, CtxTryIntoTensor};
use crate::common::{ConnectFourAction, ConnectFourBinaryObservation, ConnectFourDomain, ConnectFourPlayer};

#[derive(Clone, Debug)]
pub struct ConnectFourInfoSet{
    id: ConnectFourPlayer,
    latest_observation: ConnectFourBinaryObservation
}

impl ConnectFourInfoSet{
    pub fn new(id: ConnectFourPlayer) -> Self{
        Self{
            id, latest_observation: Default::default()
        }
    }
}

impl InformationSet<ConnectFourDomain> for ConnectFourInfoSet{
    fn agent_id(&self) -> &ConnectFourPlayer {
        &self.id
    }

    fn is_action_valid(&self, action: &ConnectFourAction) -> bool {
        self.latest_observation.board[0][action.index()] == [0,0]
    }

    fn update(&mut self, update: <ConnectFourDomain as DomainParameters>::UpdateType) -> Result<(), <ConnectFourDomain as DomainParameters>::GameErrorType> {
        self.latest_observation = update;
        Ok(())
    }
}

impl Renew<ConnectFourDomain, ()> for ConnectFourInfoSet{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ConnectFourDomain>> {
        self.latest_observation = ConnectFourBinaryObservation::default();
        Ok(())
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct ConnectFourTensorReprD1{

}

impl ConversionToTensor for ConnectFourTensorReprD1{
    fn desired_shape(&self) -> &[i64] {
        &[84]
    }
}

impl CtxTryIntoTensor<ConnectFourTensorReprD1> for ConnectFourInfoSet{
    fn try_to_tensor(&self, way: &ConnectFourTensorReprD1) -> Result<Tensor, TensorRepresentationError> {

        let mut vec = Vec::with_capacity(way.desired_shape()[0] as usize);
        for r in 0..self.latest_observation.board.len(){
            for c in 0..self.latest_observation.board[r].len(){
                vec.extend(self.latest_observation.board[r][c].map(|u| u as f32));
            }
        }
        Tensor::f_from_slice(&vec[..]).map_err(|e| TensorRepresentationError::Torch { source: e, context: format!("Converting information set: {:?}", &self) })

    }
}