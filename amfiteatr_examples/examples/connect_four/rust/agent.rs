use amfiteatr_core::agent::InformationSet;
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::error::{AmfiteatrError, ConvertError};
use amfiteatr_rl::error::TensorRepresentationError;
use amfiteatr_rl::MaskingInformationSetAction;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::{ConversionToTensor, ContextTryIntoTensor, ContextTryFromTensor, ConversionFromTensor, ContextTryIntoIndexI64, ConversionToIndexI64};
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

impl ContextTryIntoTensor<ConnectFourTensorReprD1> for ConnectFourInfoSet{
    fn try_to_tensor(&self, way: &ConnectFourTensorReprD1) -> Result<Tensor, ConvertError> {

        let mut vec = Vec::with_capacity(way.desired_shape()[0] as usize);
        for r in 0..self.latest_observation.board.len(){
            for c in 0..self.latest_observation.board[r].len(){
                vec.extend(self.latest_observation.board[r][c].map(|u| u as f32));
            }
        }
        Tensor::f_from_slice(&vec[..]).map_err(|e| ConvertError::TorchStr { origin: format!("Converting information set: {:?} ({e})", &self) })

    }
}

pub struct ConnectFourActionTensorRepresentation{}

impl ConversionFromTensor for ConnectFourActionTensorRepresentation{
    fn expected_input_shape(&self) -> &[i64] {
        &[1]
    }
}

impl ConversionToIndexI64 for ConnectFourActionTensorRepresentation{
    fn min(&self) -> i64 {
        0
    }

    fn limit(&self) -> i64 {
        6
    }
}

impl ContextTryFromTensor<ConnectFourActionTensorRepresentation> for ConnectFourAction{
    fn try_from_tensor(tensor: &Tensor, _way: &ConnectFourActionTensorRepresentation) -> Result<Self, ConvertError>
    where
        Self: Sized
    {
        let action_index = tensor.f_int64_value(&[0])
            .map_err(|e| ConvertError::ConvertFromTensor {
                origin: format!("{e:}"),
                context: format!("Converting Connect Four action from Tensor {tensor:?}")}
            )?;

        Self::try_from(action_index)
    }
}

impl ContextTryIntoIndexI64<ConnectFourActionTensorRepresentation> for ConnectFourAction{
    fn try_to_index(&self, _way: &ConnectFourActionTensorRepresentation) -> Result<i64, ConvertError> {
        Ok(self.index() as i64)
    }
}

impl MaskingInformationSetAction<ConnectFourDomain, ConnectFourActionTensorRepresentation> for ConnectFourInfoSet{
    fn try_build_mask(&self, _ctx: &ConnectFourActionTensorRepresentation) -> Result<Tensor, AmfiteatrError<ConnectFourDomain>> {
        let top_row_is_empty: [f32; 7] =  self.latest_observation.board[0].map(|[u1,u2]| {
            match (u1,u2) {
                (0,0) => 1.0,
                _ => 0.0,
            }
        });
        Ok(Tensor::f_from_slice(&top_row_is_empty)?)

    }
}