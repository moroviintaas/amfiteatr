use amfiteatr_core::agent::InformationSet;
use amfiteatr_core::error::ConvertError;
use amfiteatr_core::scheme::Scheme;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::{ContextEncodeTensor, TensorEncoding};
use crate::cart_pole::common::{CartPoleObservation, CartPoleScheme, SINGLE_PLAYER_ID};

impl InformationSet<CartPoleScheme> for CartPoleObservation{
    fn agent_id(&self) -> &<CartPoleScheme as Scheme>::AgentId {
        &SINGLE_PLAYER_ID
    }

    fn is_action_valid(&self, action: &<CartPoleScheme as Scheme>::ActionType) -> bool {
        true
    }

    fn update(&mut self, update: <CartPoleScheme as Scheme>::UpdateType) -> Result<(), <CartPoleScheme as Scheme>::GameErrorType> {
        self.angle = update.angle;
        self.position = update.position;
        self.angular_velocity = update.angular_velocity;
        self.velocity = update.velocity;

        Ok(())
    }
}


pub struct CartPoleObservationEncoding{

}

impl TensorEncoding for CartPoleObservationEncoding{
    fn desired_shape(&self) -> &[i64] {
        &[4]
    }
}

impl ContextEncodeTensor<CartPoleObservationEncoding> for CartPoleObservation{
    fn try_to_tensor(&self, _encoding: &CartPoleObservationEncoding) -> Result<Tensor, ConvertError> {
        let data = [self.position, self.velocity, self.angle, self.angular_velocity];
        Ok(Tensor::from_slice(&data))
    }
}