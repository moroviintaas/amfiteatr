use std::fs::File;
use std::marker::PhantomData;
use std::ops::Div;
use rand::distr::Distribution;
use amfiteatr_core::agent::{InformationSet, Policy};
use amfiteatr_core::domain::{DomainParameters, Reward};
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


pub struct GeneticPolicyGeneric<DP: DomainParameters, S: PolicySpecimen<DP, M> + Sized, M: Send>
{

    config: GeneticPolicyConfig,
    population: Vec<S>,
    tboard_writer: Option<tboard::EventWriter<File>>,
    selected_specimen_index: usize,
    saved_payoffs: Vec<Vec<DP::UniversalReward>>,
    _mutagen: PhantomData<M>
}



impl <DP: DomainParameters, S: PolicySpecimen<DP, M>, M: Send> GeneticPolicyGeneric<DP, S, M>
{


    pub fn new<G>(config: GeneticPolicyConfig, generator: G) -> GeneticPolicyGeneric<DP, S, M>
    where G: Fn(usize) -> S{
        /*let d = StandardUniform::default();
        let population: Vec<S> = (0..config.target_population)
            .map(|_| d.sample(&mut rand::rng())).collect();//vec![d.sample(&mut rand::rng());config.target_population];
        */
        let population: Vec<S> = (0..config.target_population)
            .map(|i| generator(i)).collect();//vec![d.sample(&mut rand::rng());config.target_population];
        let saved_payoffs: Vec<Vec<DP::UniversalReward>> = vec![Vec::new(); population.len()];
        Self{
            population,
            config,
            saved_payoffs,
            tboard_writer: None,
            selected_specimen_index: 0,
            _mutagen: Default::default(),
        }
    }

    pub fn update_policy(&mut self) -> Result<(), AmfiteatrError<DP>>
    where <DP as DomainParameters>::UniversalReward: Div<f64, Output=DP::UniversalReward>
    {
    //where for<'a> &'a DP::UniversalReward: Sum + Div<f64> {
        let _averages: Vec<DP::UniversalReward>  = self.saved_payoffs.iter()
            //.map(|v|v.iter().sum::<DP::UniversalReward>()/self.population.len() as f64).collect();
            .map(|v|{
                v.iter().fold(<DP::UniversalReward as Reward>::neutral(), |acc, x|{
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
    DP: DomainParameters,
    IS: InformationSet<DP>,
    SM: PolicySpecimen<DP, M> + Policy<DP, InfoSetType=IS> + Sized,
    M: Send
> Policy<DP> for GeneticPolicyGeneric<DP, SM, M>
{
    type InfoSetType = IS;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        let sub_policy = self.population.get(self.selected_specimen_index)
            .ok_or(AmfiteatrError::Custom(format!("Failed selecting genetic subpolicy with index: {}", self.selected_specimen_index)))?;

        sub_policy.select_action(state)
    }

    fn call_on_episode_start(&mut self) -> Result<(), AmfiteatrError<DP>> {
        let index = rand::distr::Uniform::new(0, self.population.len())
            .map_err(|e| AmfiteatrError::Custom("Creating index distribution for specimen {e}".into()))?
            .sample(&mut rand::rng());
        self.selected_specimen_index = index;
        Ok(())
    }

    fn call_on_episode_finish(&mut self, final_env_reward: DP::UniversalReward) -> Result<(), AmfiteatrError<DP>> {
        self.saved_payoffs[self.selected_specimen_index].push(final_env_reward);

        Ok(())
    }

    fn call_between_epochs(&mut self) -> Result<(), AmfiteatrError<DP>> {
        for s in self.saved_payoffs.iter_mut(){
            s.clear();
        }
        Ok(())
    }
}

