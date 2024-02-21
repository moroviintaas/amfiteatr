use pyo3::{pymethods, PyResult};
use amfiteatr_core::agent::InformationSet;
use amfiteatr_core::domain::DomainParameters;
use crate::common::{CartPoleDomain, CartPoleObservation, SINGLE_PLAYER_ID};

#[derive(Debug, Clone)]
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

    fn is_action_valid(&self, action: &<CartPoleDomain as DomainParameters>::ActionType) -> bool {
        match action{
            0 | 1 => true,
            _ => false
        }
    }

    fn update(&mut self, update: <CartPoleDomain as DomainParameters>::UpdateType)
        -> Result<(), <CartPoleDomain as DomainParameters>::GameErrorType> {

        self.latest_observation = update;
        Ok(())

    }
}