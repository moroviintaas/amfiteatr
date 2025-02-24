use std::cmp::Ordering;
use std::marker::PhantomData;
use rand::{Rng, thread_rng};
use amfiteatr_core::agent::{InformationSet, Policy};
use amfiteatr_core::error::AmfiteatrError;
use crate::domain::{ClassicAction, ClassicGameDomain, ClassicGameError, UsizeAgentId};
use crate::domain::ClassicAction::{Down, Up};

/// Classic pure strategy - allways one specified action from [`ClassicAction`].
pub struct ClassicPureStrategy<ID: UsizeAgentId, IS: InformationSet<ClassicGameDomain<ID>>>{
    pub action: ClassicAction,
    _is: PhantomData<IS>,
    _id: PhantomData<ID>,
}

impl<ID: UsizeAgentId, IS: InformationSet<ClassicGameDomain<ID>>> ClassicPureStrategy<ID, IS>{
    pub fn new(action: ClassicAction) -> Self{
        Self{
            action,
            _is: Default::default(),
            _id: Default::default()
        }
    }



}
impl<ID: UsizeAgentId, IS: InformationSet<ClassicGameDomain<ID>>> Policy<ClassicGameDomain<ID>> for ClassicPureStrategy<ID, IS>{
    type InfoSetType = IS ;

    fn select_action(&self, _state: &Self::InfoSetType) -> Result<ClassicAction, AmfiteatrError<ClassicGameDomain<ID>>> {
        Ok(self.action)
    }
}
/// Selects action [`Up`] with given probability, otherwise [`Down`].
pub struct ClassicMixedStrategy<ID: UsizeAgentId, IS: InformationSet<ClassicGameDomain<ID>>>{
    probability_up: f64,
    _is: PhantomData<IS>,
    _id: PhantomData<ID>,
}

impl<ID: UsizeAgentId, IS: InformationSet<ClassicGameDomain<ID>>> ClassicMixedStrategy<ID, IS>{
    pub fn new(probability_up: f64) -> Self{
        Self{
            probability_up,
            _is: Default::default(),
            _id: Default::default(),
        }
    }
    pub fn new_checked(probability: f64) -> Result<Self, ClassicGameError<ID>>{
        if !(0.0..=1.0).contains(&probability){
            Err(ClassicGameError::NotAProbability(probability))
        } else{
            Ok(Self::new(probability))
        }
    }
}

impl<ID: UsizeAgentId, IS: InformationSet<ClassicGameDomain<ID>>> Policy<ClassicGameDomain<ID>> for ClassicMixedStrategy<ID, IS>{
    type InfoSetType = IS ;

    fn select_action(&self, _state: &Self::InfoSetType) -> Result<ClassicAction, AmfiteatrError<ClassicGameDomain<ID>>> {
        let mut rng = thread_rng();
        let sample = rng.gen_range(0.0..1.0);
        sample.partial_cmp(&self.probability_up).map(|o|{
          match o{
              Ordering::Less => Up,
              Ordering::Equal => Down,
              Ordering::Greater => Down,
          }
        }).ok_or(AmfiteatrError::NoActionAvailable {
            context: "ClassicMixedStrategy".to_string(),
        })

    }
}

