use log::debug;
use sha2::{Digest, Sha256};
use amfiteatr_core::agent::{InformationSet, PresentPossibleActions};
use amfiteatr_core::demo::DemoAction;
use amfiteatr_core::scheme::{Scheme, Renew};
use amfiteatr_core::error::AmfiteatrError;
use crate::expensive_update::scheme::ExpensiveUpdateScheme;
#[derive(Debug, Clone)]
pub struct ExpensiveUpdateInformationSet{
    id: u64,

}

impl ExpensiveUpdateInformationSet{
    pub fn new(id: u64) -> ExpensiveUpdateInformationSet{
        Self{id}
    }
}


impl InformationSet<ExpensiveUpdateScheme> for ExpensiveUpdateInformationSet{
    fn agent_id(&self) -> &<ExpensiveUpdateScheme as Scheme>::AgentId {
        &self.id
    }

    fn is_action_valid(&self, _action: &<ExpensiveUpdateScheme as Scheme>::ActionType) -> bool {
        true
    }

    fn update(&mut self, update: <ExpensiveUpdateScheme as Scheme>::UpdateType) -> Result<(), <ExpensiveUpdateScheme as Scheme>::GameErrorType> {

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

impl PresentPossibleActions<ExpensiveUpdateScheme> for ExpensiveUpdateInformationSet{
    type ActionIteratorType = Vec<DemoAction>;

    fn available_actions(&self) -> Self::ActionIteratorType {
        vec![DemoAction(0)]
    }
}

impl Renew<ExpensiveUpdateScheme, ()> for ExpensiveUpdateInformationSet{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ExpensiveUpdateScheme>> {
        Ok(())
    }
}