use log::debug;
use sha2::{Digest, Sha256};
use amfiteatr_core::agent::{InformationSet, PresentPossibleActions};
use amfiteatr_core::demo::DemoAction;
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::error::AmfiteatrError;
use crate::expensive_update::domain::ExpensiveUpdateDomain;
#[derive(Debug, Clone)]
pub struct ExpensiveUpdateInformationSet{
    id: u64,

}

impl ExpensiveUpdateInformationSet{
    pub fn new(id: u64) -> ExpensiveUpdateInformationSet{
        Self{id}
    }
}


impl InformationSet<ExpensiveUpdateDomain> for ExpensiveUpdateInformationSet{
    fn agent_id(&self) -> &<ExpensiveUpdateDomain as DomainParameters>::AgentId {
        &self.id
    }

    fn is_action_valid(&self, _action: &<ExpensiveUpdateDomain as DomainParameters>::ActionType) -> bool {
        true
    }

    fn update(&mut self, update: <ExpensiveUpdateDomain as DomainParameters>::UpdateType) -> Result<(), <ExpensiveUpdateDomain as DomainParameters>::GameErrorType> {

        let mut h = Vec::new();
        h.extend(self.id.to_be_bytes());
        debug!("Agent: {}, expensive update of cost {}", self.id, update);
        for _ in 0..update{
            let n = Sha256::digest(&h[..]);
            h.clear();
            h.extend_from_slice(n.as_slice())
        }
        Ok(())
    }
}

impl PresentPossibleActions<ExpensiveUpdateDomain> for ExpensiveUpdateInformationSet{
    type ActionIteratorType = Vec<DemoAction>;

    fn available_actions(&self) -> Self::ActionIteratorType {
        vec![DemoAction(0)]
    }
}

impl Renew<ExpensiveUpdateDomain, ()> for ExpensiveUpdateInformationSet{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ExpensiveUpdateDomain>> {
        Ok(())
    }
}