use std::fs::File;
use std::marker::PhantomData;
use std::ops::Div;
use rand::distr::Distribution;
use amfiteatr_core::agent::{InformationSet, Policy};
use amfiteatr_core::scheme::{Scheme, Reward};
use amfiteatr_core::error::AmfiteatrError;
use crate::policy::PolicySpecimen;

pub struct GeneticPolicyConfig{
    pub target_population: usize,

    pub group_alpha_population: usize,
    pub group_beta_population: usize,



    pub mutation_probability: f64,

    pub breed_alpha_alpha_number: usize,

    pub breed_alpha_beta_number: usize,

}


pub struct GeneticPolicyGeneric<S: Scheme, SP: PolicySpecimen<S, M> + Sized, M: Send>
{

    config: GeneticPolicyConfig,
    population: Vec<SP>,
    tboard_writer: Option<tboard::EventWriter<File>>,
    selected_specimen_index: usize,
    saved_payoffs: Vec<Vec<S::UniversalReward>>,
    _mutagen: PhantomData<M>
}



impl <S: Scheme, PS: PolicySpecimen<S, M>, M: Send> GeneticPolicyGeneric<S, PS, M>
{


    pub fn new<G>(config: GeneticPolicyConfig, generator: G) -> GeneticPolicyGeneric<S, PS, M>
    where G: Fn(usize) -> PS{
        /*let d = StandardUniform::default();
        let population: Vec<S> = (0..config.target_population)
            .map(|_| d.sample(&mut rand::rng())).collect();//vec![d.sample(&mut rand::rng());config.target_population];
        */
        let population: Vec<PS> = (0..config.target_population)
            .map(|i| generator(i)).collect();//vec![d.sample(&mut rand::rng());config.target_population];
        let saved_payoffs: Vec<Vec<S::UniversalReward>> = vec![Vec::new(); population.len()];
        Self{
            population,
            config,
            saved_payoffs,
            tboard_writer: None,
            selected_specimen_index: 0,
            _mutagen: Default::default(),
        }
    }

    pub fn update_policy(&mut self) -> Result<(), AmfiteatrError<S>>
    where <S as Scheme>::UniversalReward: Div<f64, Output=S::UniversalReward>
    {
    //where for<'a> &'a S::UniversalReward: Sum + Div<f64> {
        let _averages: Vec<S::UniversalReward>  = self.saved_payoffs.iter()
            //.map(|v|v.iter().sum::<S::UniversalReward>()/self.population.len() as f64).collect();
            .map(|v|{
                v.iter().fold(<S::UniversalReward as Reward>::neutral(), |acc, x|{
                    acc+x
                })/ self.population.len() as f64
            }).collect();

        todo!();
        for s in self.saved_payoffs.iter_mut(){
            s.clear();
        }
        Ok(())
    }
}

impl <
    S: Scheme,
    IS: InformationSet<S>,
    SM: PolicySpecimen<S, M> + Policy<S, InfoSetType=IS> + Sized,
    M: Send
> Policy<S> for GeneticPolicyGeneric<S, SM, M>
{
    type InfoSetType = IS;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        let sub_policy = self.population.get(self.selected_specimen_index)
            .ok_or(AmfiteatrError::Custom(format!("Failed selecting genetic subpolicy with index: {}", self.selected_specimen_index)))?;

        sub_policy.select_action(state)
    }

    fn call_on_episode_start(&mut self) -> Result<(), AmfiteatrError<S>> {
        let index = rand::distr::Uniform::new(0, self.population.len())
            .map_err(|e| AmfiteatrError::Custom("Creating index distribution for specimen {e}".into()))?
            .sample(&mut rand::rng());
        self.selected_specimen_index = index;
        Ok(())
    }

    fn call_on_episode_finish(&mut self, final_env_reward: S::UniversalReward) -> Result<(), AmfiteatrError<S>> {
        self.saved_payoffs[self.selected_specimen_index].push(final_env_reward);

        Ok(())
    }

    fn call_between_epochs(&mut self) -> Result<(), AmfiteatrError<S>> {
        for s in self.saved_payoffs.iter_mut(){
            s.clear();
        }
        Ok(())
    }
}

