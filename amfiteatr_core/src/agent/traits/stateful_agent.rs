use crate::agent::IdAgent;
use crate::agent::info_set::InformationSet;
use crate::domain::DomainParameters;

/// Agent that holds some game state
/// > Formally agent knows _information set_ which can be described as state of the game
/// from point of view of the agent.
pub trait StatefulAgent<DP: DomainParameters>{
    type InfoSetType: InformationSet<DP>;

    /// Updated underlying information set using domain's updated type.
    fn update(&mut self, info_set_update: DP::UpdateType) -> Result<(), DP::GameErrorType>;
    /// Return reference to underlying information set.
    fn info_set(&self) -> &Self::InfoSetType;



}

impl<DP:DomainParameters, T: StatefulAgent<DP>> IdAgent<DP> for T{
    fn id(&self) -> &<DP as DomainParameters>::AgentId {
        self.info_set().agent_id()
    }
}
/*
impl<DP: DomainParameters, T: StatefulAgent<DP>> StatefulAgent<DP> for Arc<Mutex<T>> {
    type InfoSetType = T::InfoSetType;

    fn update(&mut self, info_set_update: DP::UpdateType) -> Result<(), DP::GameErrorType> {
        let mut g = self.lock().unwrap();
        g.update(info_set_update)
    }

    fn info_set(&self) -> &Self::InfoSetType {
        let mut g = self.lock().unwrap();
        g.info_set()
    }
}*/