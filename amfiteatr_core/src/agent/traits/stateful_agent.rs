use crate::agent::IdAgent;
use crate::agent::info_set::InformationSet;
use crate::scheme::Scheme;

/// Agent that holds some game state
/// > Formally agent knows _information set_ which can be described as state of the game
/// > from point of view of the agent.
pub trait StatefulAgent<S: Scheme>{
    type InfoSetType: InformationSet<S>;

    /// Updated underlying information set using scheme's updated type.
    fn update(&mut self, info_set_update: S::UpdateType) -> Result<(), S::GameErrorType>;
    /// Return reference to underlying information set.
    fn info_set(&self) -> &Self::InfoSetType;



}

impl<S: Scheme, T: StatefulAgent<S>> IdAgent<S> for T{
    fn id(&self) -> &<S as Scheme>::AgentId {
        self.info_set().agent_id()
    }
}
/*
impl<S: DomainParameters, T: StatefulAgent<S>> StatefulAgent<S> for Arc<Mutex<T>> {
    type InfoSetType = T::InfoSetType;

    fn update(&mut self, info_set_update: S::UpdateType) -> Result<(), S::GameErrorType> {
        let mut g = self.lock().unwrap();
        g.update(info_set_update)
    }

    fn info_set(&self) -> &Self::InfoSetType {
        let mut g = self.lock().unwrap();
        g.info_set()
    }
}*/