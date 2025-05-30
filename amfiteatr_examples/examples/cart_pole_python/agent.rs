use amfiteatr_core::agent::{InformationSet, PresentPossibleActions};
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::error::{AmfiteatrError, ConvertError};
use amfiteatr_proc_macro::no_assessment_info_set;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::{TensorEncoding, ContextEncodeTensor};
use crate::common::{CartPoleAction, CartPoleDomain, CartPoleObservation, SINGLE_PLAYER_ID};


#[derive(Debug, Clone, Default)]
#[no_assessment_info_set(CartPoleDomain)]
pub struct PythonGymnasiumCartPoleInformationSet{
    latest_observation: CartPoleObservation
}

impl PythonGymnasiumCartPoleInformationSet{

    pub fn new( initial_observation: CartPoleObservation)
        -> Self{
        Self{
            latest_observation: initial_observation
        }
    }
}

impl InformationSet<CartPoleDomain> for PythonGymnasiumCartPoleInformationSet{
    fn agent_id(&self) -> &<CartPoleDomain as DomainParameters>::AgentId {
        &SINGLE_PLAYER_ID
    }

    fn is_action_valid(&self, _action: &<CartPoleDomain as DomainParameters>::ActionType) -> bool {
        true
    }

    fn update(&mut self, update: <CartPoleDomain as DomainParameters>::UpdateType)
        -> Result<(), <CartPoleDomain as DomainParameters>::GameErrorType> {

        self.latest_observation = update;
        Ok(())
    }
}



impl Renew<CartPoleDomain, CartPoleObservation> for PythonGymnasiumCartPoleInformationSet{
    fn renew_from(&mut self, base: CartPoleObservation) -> Result<(), AmfiteatrError<CartPoleDomain>> {
        self.latest_observation = base;
        Ok(())
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct CartPoleInformationSetConversion{}
pub const CART_POLE_TENSOR_REPR: CartPoleInformationSetConversion = CartPoleInformationSetConversion{};



impl TensorEncoding for CartPoleInformationSetConversion{
    fn desired_shape(&self) -> &[i64] {
        &[4]
    }
}

impl ContextEncodeTensor<CartPoleInformationSetConversion> for PythonGymnasiumCartPoleInformationSet{
    fn try_to_tensor(&self, _way: &CartPoleInformationSetConversion) -> Result<Tensor, ConvertError> {
        let v = vec![
            self.latest_observation.position,
            self.latest_observation.velocity,
            self.latest_observation.angle,
            self.latest_observation.angular_velocity];

        Tensor::f_from_slice(&v)
            .map_err(|e| ConvertError::TorchStr {
                origin: format!("Failed to convert observation to tensor: {e}"),
            })

    }
}
/*
impl EvaluatedInformationSet<CartPoleDomain> for PythonGymnasiumCartPoleInformationSet{
    type RewardType = NoneReward;

    fn current_subjective_score(&self) -> Self::RewardType {
        NoneReward{}
    }

    fn penalty_for_illegal(&self) -> Self::RewardType {
        NoneReward{}
    }
}

 */


impl PresentPossibleActions<CartPoleDomain> for PythonGymnasiumCartPoleInformationSet{
    type ActionIteratorType = [CartPoleAction;2];

    fn available_actions(&self) -> Self::ActionIteratorType {
        [CartPoleAction::Left, CartPoleAction::Right]
    }
}