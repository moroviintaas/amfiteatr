mod options;

use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use log::{
    debug,
    info,
};
use amfiteatr_rl::tch::{nn, Device, Tensor};
use clap::Parser;
use plotters::style::colors;
use amfiteatr_rl::tch::nn::{Adam, VarStore};
use amfiteatr_core::agent::*;
use amfiteatr_core::comm::{
    AgentMpscAdapter,
    EnvironmentMpscPort
};
use amfiteatr_core::env::{AutoEnvironmentWithScores, ReseedEnvironment, TracingBasicEnvironment, TracingEnvironment};
use amfiteatr_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfiteatr_core::agent::RewardedAgent;
use amfiteatr_core::agent::TracingAgent;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_classic::domain::{
    AgentNum,
    ClassicAction,
    ClassicGameDomainNumbered
};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::{AsymmetricRewardTableInt, SymmetricRewardTable};
use amfiteatr_classic::agent::{
    LocalHistoryConversionToTensor,
    LocalHistoryInfoSet};
use amfiteatr_examples::plots::{plot_many_series, PlotSeries};
use amfiteatr_examples::series::PayoffGroupSeries;
use amfiteatr_rl::policy::{ActorCriticPolicy, LearningNetworkPolicyGeneric, TrainConfig};
use amfiteatr_rl::tensor_data::TensorEncoding;
use amfiteatr_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorActorCritic};
use crate::options::ReplicatorOptions;


pub fn avg(entries: &[f32]) -> Option<f32>{
    if entries.is_empty(){
        None
    } else {
        let sum = entries.iter().sum::<f32>();
        Some(sum / entries.len() as f32)
    }
}

pub fn setup_logger(options: &ReplicatorOptions) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        //.level(options.general_log_level)
        .level_for("amfiteatr_rl", options.rl_log_level)
        .level_for("replicator_dynamics_old", options.log_level)
        .level_for("amfiteatr_classic", options.classic_log_level)
        .level_for("amfiteatr_core", options.log_level_amfi);

        match &options.log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        .apply()?;
    Ok(())
}
type D = ClassicGameDomainNumbered;
type S = PairingState<<D as DomainParameters>::AgentId>;
type Pol = ActorCriticPolicy<D, LocalHistoryInfoSet<<D as DomainParameters>::AgentId>, LocalHistoryConversionToTensor>;
type MixedPolicy = ClassicMixedStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>;
type PurePolicy = ClassicPureStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>;
type AgentComm = AgentMpscAdapter<D>;

pub enum Group{
    Mixes,
    Hawks,
    Doves,
    Learning
}

struct Model{
    pub environment:  TracingBasicEnvironment<D, S, EnvironmentMpscPort<D>>,
    //agents: Arc<Mutex<dyn MultiEpisodeAgent<D, (), InfoSetType=()>>>,
    pub mixed_agents: Vec<Arc<Mutex<AgentGen<D, MixedPolicy, AgentComm>>>>,
    pub hawk_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
    pub dove_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
    pub learning_agents: Vec<Arc<Mutex<TracingAgentGen<D, Pol, AgentComm>>>>,

    //averages in groups in epochs - one entry in vec is average of players in that group for that episode
    pub averages_mixed: Vec<f32>,
    pub averages_hawk: Vec<f32>,
    pub averages_dove: Vec<f32>,
    pub averages_learning: Vec<f32>,
    pub averages_all: Vec<f32>,

    pub average_learning_defects: Vec<f32>,
    pub average_learning_coops: Vec<f32>,

    learning_defects: Vec<f32>,
    learning_coops: Vec<f32>,

    scores_mixed: Vec<f32>,
    scores_hawk: Vec<f32>,
    scores_dove: Vec<f32>,
    scores_learning: Vec<f32>,
    scores_all: Vec<f32>,
}

impl Model{

    #[allow(dead_code)]
    pub fn new(environment: TracingBasicEnvironment<D, S, EnvironmentMpscPort<D>>) -> Self{
        Self{
            environment, mixed_agents: Vec::new(), hawk_agents: Vec::new(), dove_agents: Vec::new(),
            learning_agents: Vec::new(), averages_mixed: Vec::new(),
            averages_hawk: Vec::new(),
            averages_dove: Vec::new(),
            averages_learning: Vec::new(),
            averages_all: Vec::new(),
            average_learning_defects: vec![],
            average_learning_coops: vec![],
            learning_defects: vec![],
            learning_coops: vec![],
            scores_mixed: vec![],
            scores_hawk: vec![],
            scores_dove: vec![],
            scores_learning: vec![],
            scores_all: vec![],
        }
    }

    pub fn new_with_agents(environment: TracingBasicEnvironment<D, S, EnvironmentMpscPort<D>>,
                           learning_agents: Vec<Arc<Mutex<TracingAgentGen<D, Pol, AgentComm>>>>,
                           mixed_agents: Vec<Arc<Mutex<AgentGen<D, MixedPolicy, AgentComm>>>>,
                           hawk_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
                           dove_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
        ) -> Self{
        Self{
            environment, learning_agents,
            mixed_agents, hawk_agents, dove_agents,
            averages_mixed: vec![],
            averages_hawk: vec![],
            averages_dove: vec![],
            averages_learning: vec![],
            averages_all: vec![],
            average_learning_defects: vec![],
            average_learning_coops: vec![],
            learning_defects: vec![],
            learning_coops: vec![],
            scores_mixed: vec![],
            scores_hawk: vec![],
            scores_dove: vec![],
            scores_learning: vec![],


            scores_all: vec![],
        }

    }
    pub fn clear_averages(&mut self){
        self.averages_dove.clear();
        self.averages_hawk.clear();
        self.averages_learning.clear();
        self.averages_all.clear();
        self.averages_mixed.clear();
        self.average_learning_coops.clear();
        self.average_learning_defects.clear();
    }

    #[allow(dead_code)]
    pub fn clear_trajectories(&mut self){

        for agent in &self.learning_agents{
            let mut guard = agent.lock().unwrap();
            guard.reset_trajectory()
        }

    }

    pub fn remember_average_group_scores(&mut self){
        self.clear_episode_scores();

        for agent in &self.learning_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f32;
            let coops = guard.episodes().last().map(|t| t.iter().filter(|t|{
                t.action() == &ClassicAction::Down
            }).count()).unwrap_or(0usize);
            /*
            let defects = guard.game_trajectory().list().iter().filter(|t|{
                t.taken_action() == &ClassicAction::Defect
            }).count();


             */
            let defects = guard.episodes().last().map(|t| t.iter().filter(|t|{
                t.action() == &ClassicAction::Up
            }).count()).unwrap_or(0usize);
            self.learning_defects.push(defects as f32);
            self.learning_coops.push(coops as f32);
            self.scores_all.push(score);
            self.scores_learning.push(score);
        }
        for agent in &self.mixed_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f32;
            self.scores_all.push(score);
            self.scores_mixed.push(score);
        }
        for agent in &self.dove_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f32;
            self.scores_all.push(score);
            self.scores_dove.push(score);
        }
        for agent in &self.hawk_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f32;
            self.scores_all.push(score);
            self.scores_hawk.push(score);
        }

        if let Some(average) = avg(&self.scores_dove[..]){
            self.averages_dove.push(average)
        }
        if let Some(average) = avg(&self.scores_hawk[..]){
            self.averages_hawk.push(average)
        }
        if let Some(average) = avg(&self.scores_mixed[..]){
            self.averages_mixed.push(average)
        }
        if let Some(average) = avg(&self.scores_learning[..]){
            self.averages_learning.push(average)
        }
        if let Some(average) = avg(&self.scores_all[..]){
            self.averages_all.push(average)
        }

        if let Some(average) = avg(&self.learning_coops[..]){
            self.average_learning_coops.push(average)
        }
        if let Some(average) = avg(&self.learning_defects[..]){
            debug!("Average defect number in round: {} ", average);
            self.average_learning_defects.push(average)
        }

    }

    pub fn clear_episode_scores(&mut self){
        self.scores_mixed.clear();
        self.scores_all.clear();
        self.scores_dove.clear();
        self.scores_hawk.clear();
        self.scores_learning.clear();
        self.learning_coops.clear();
        self.learning_defects.clear();
    }

    pub fn run_episode(&mut self) -> Result<(), AmfiteatrError<D>>{

        thread::scope(|s|{
            s.spawn(||{
                self.environment.reseed(()).unwrap();
                self.environment.run_with_scores().unwrap();
            });
            for a in  &self.dove_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode(()).unwrap();

                });
            };

            for a in  &self.hawk_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode(()).unwrap();

                });
            };

            for a in  &self.mixed_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode(()).unwrap();

                });
            };

            for a in  &self.learning_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode(()).unwrap();

                });
            };



        });

        Ok(())
    }

    pub fn update_policies(&mut self) -> Result<(), AmfiteatrError<D>>{
        for a in &self.learning_agents{
            let mut agent = a.lock().unwrap();
            let trajectories = agent.take_episodes();
            agent.policy_mut().train_on_trajectories_env_reward(&trajectories[..])?;
        }
        Ok(())
    }
}


fn main() -> Result<(), AmfiteatrError<D>>{
    debug!("Starting");

    let args = ReplicatorOptions::parse();
    setup_logger(&args).unwrap();
    let device = Device::Cpu;
    //let device = Device::Cpu;

    let reward_table: AsymmetricRewardTableInt =
        SymmetricRewardTable::new(2, 1, 4, 0).into();
    //let env_state_template = PairingState::new_even(number_of_players, args.number_of_rounds, reward_table).unwrap();
    let tensor_repr = LocalHistoryConversionToTensor::new(args.number_of_rounds);
    let input_size = tensor_repr.desired_shape().iter().product();
    //let mut comms = HashMap::<u32, SyncCommEnv<ClassicGameDomainNumbered>>::with_capacity(number_of_players);

    let net_template = NeuralNetTemplate::new(|path|{
        let seq = nn::seq()
            .add(nn::linear(path / "input", input_size, 512, Default::default()))
            //.add(nn::linear(path / "h1", 256, 256, Default::default()))
            .add(nn::linear(path / "hidden1", 512, 512, Default::default()))
            .add_fn(|xs|xs.relu());
            //.add(nn::linear(path / "h2", 512, 512, Default::default()));
        let actor = nn::linear(path / "al", 512, 2, Default::default());
        let critic =  nn::linear(path / "ac", 512, 1, Default::default());
        {move |input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });

    let mut env_adapter = EnvironmentMpscPort::new();

    let mut learning_agents: Vec<Arc<Mutex<TracingAgentGen<D, Pol, AgentComm>>>> = Vec::new();
    let mut mixed_agents: Vec<Arc<Mutex<AgentGen<D, MixedPolicy, AgentComm>>>> = Vec::new();
    let mut hawk_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>> = Vec::new();
    let mut dove_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>> = Vec::new();

    let mut report_average_hawk_reward = Vec::with_capacity(args.epochs + 1);
    let mut report_average_dove_reward = Vec::with_capacity(args.epochs + 1);
    let mut report_average_mixed_reward = Vec::with_capacity(args.epochs + 1);
    let mut report_average_all_reward = Vec::with_capacity(args.epochs + 1);
    let mut report_average_learning_reward = Vec::with_capacity(args.epochs + 1);
    let mut report_average_coops = Vec::with_capacity(args.epochs + 1);
    let mut report_average_defects = Vec::with_capacity(args.epochs + 1);

    let offset_learning = 0 as AgentNum;
    let offset_mixed = args.number_of_learning as AgentNum;
    let offset_hawk = args.number_of_mixes as AgentNum + offset_mixed;
    let offset_dove = args.number_of_hawks as AgentNum + offset_hawk;
    let total_number_of_players = offset_dove as usize + args.number_of_doves;

    for i in offset_learning..offset_mixed{
        let comm = env_adapter.register_agent(i)?;
        let state = LocalHistoryInfoSet::new(i, reward_table);
        let net = A2CNet::new(VarStore::new(device), net_template.get_net_closure());
        let opt = net.build_optimizer(Adam::default(), 1e-4).unwrap();
        let policy = ActorCriticPolicy::new(net, opt, tensor_repr, TrainConfig {gamma: 0.99});
        let agent = TracingAgentGen::new(state, comm, policy);
        learning_agents.push(Arc::new(Mutex::new(agent)));

    }
    debug!("Created learning agent vector");

    for i in offset_mixed..offset_hawk{
        let comm = env_adapter.register_agent(i)?;
        let state = LocalHistoryInfoSet::new(i, reward_table);

        let policy = MixedPolicy::new(args.mix_probability_of_hawk);
        let agent = AgentGen::new(state, comm, policy);
        mixed_agents.push(Arc::new(Mutex::new(agent)));
    }

    for i in offset_hawk..offset_dove{
        let comm = env_adapter.register_agent(i)?;
        let state = LocalHistoryInfoSet::new(i, reward_table);

        let policy = PurePolicy::new(ClassicAction::Up);
        let agent = AgentGen::new(state, comm, policy);
        hawk_agents.push(Arc::new(Mutex::new(agent)));

    }
    for i in offset_dove..total_number_of_players as AgentNum{
        let comm = env_adapter.register_agent(i)?;
        let state = LocalHistoryInfoSet::new(i, reward_table);

        let policy = PurePolicy::new(ClassicAction::Down);
        let agent = AgentGen::new(state, comm, policy);
        dove_agents.push(Arc::new(Mutex::new(agent)));

    }
    let env_state = PairingState::new_even(total_number_of_players,
                                           args.number_of_rounds, reward_table)?;
    let environment = TracingBasicEnvironment::new(env_state, env_adapter);


    let mut model = Model::new_with_agents(environment, learning_agents, mixed_agents,
                                           hawk_agents, dove_agents);

    // inital test

    info!("Starting initial evaluation");
    for _i in 0..100{
        model.run_episode()?;
        model.remember_average_group_scores();
    }


    if let Some(average) = avg(&model.averages_learning){
            info!("Average learning agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_learning_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_dove){
            info!("Average dove agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_dove_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_hawk){
            info!("Average hawk agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_hawk_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_mixed){
            info!("Average mixed({}) agent score in {} rounds: {:.02}", args.mix_probability_of_hawk , args.number_of_rounds, average );
            report_average_mixed_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_all){
            info!("Average any agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_all_reward.push(average);
        }
        if let Some(average) = avg(&model.average_learning_defects){
            info!("Average learning agent defected {}  in rounds: {:.02}", average, args.number_of_rounds,);
            report_average_defects.push(average);
        }
        if let Some(average) = avg(&model.average_learning_coops){
            info!("Average learning agent cooperated {}  in rounds: {:.02}", average, args.number_of_rounds,);
            report_average_coops.push(average);
        }

    for e in 0..args.epochs{
        info!("Running training epoch: {}", e);
        for _ in 0..args.batch_size{
            model.run_episode()?;
        }
        model.update_policies()?;

        info!("Testing after epoch: {}", e);
        model.clear_averages();
        for _i in 0..100{
            model.run_episode()?;
            model.remember_average_group_scores();

        }
        if let Some(average) = avg(&model.averages_learning){
            info!("Average learning agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_learning_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_dove){
            info!("Average dove agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_dove_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_hawk){
            info!("Average hawk agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_hawk_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_mixed){
            info!("Average mixed({}) agent score in {} rounds: {:.02}", args.mix_probability_of_hawk , args.number_of_rounds, average );
            report_average_mixed_reward.push(average);
        }
        if let Some(average) = avg(&model.averages_all){
            info!("Average any agent score in {} rounds: {:.02}", args.number_of_rounds, average );
            report_average_all_reward.push(average);
        }
        if let Some(average) = avg(&model.average_learning_defects){
            info!("Average learning agent defected {}  in rounds: {:.02}", average, args.number_of_rounds,);
            report_average_defects.push(average);
        }
        if let Some(average) = avg(&model.average_learning_coops){
            info!("Average learning agent cooperated {}  in rounds: {:.02}", average, args.number_of_rounds,);
            report_average_coops.push(average);
        }

    }

    let mut payoff_series = vec![];


    if !report_average_learning_reward.is_empty(){
        payoff_series.push(PayoffGroupSeries{
            id: "Learning".to_string(),
            payoffs: report_average_learning_reward.clone(),
        });
    }
    if !report_average_hawk_reward.is_empty(){
        payoff_series.push(PayoffGroupSeries{
            id: "Hawk".to_string(),
            payoffs: report_average_hawk_reward.clone(),
        });
    }
    if !report_average_dove_reward.is_empty(){
        payoff_series.push(PayoffGroupSeries{
            id: "Dove".to_string(),
            payoffs: report_average_dove_reward.clone(),
        });
    }
    if !report_average_mixed_reward.is_empty(){
        payoff_series.push(PayoffGroupSeries{
            id: "Mixed".to_string(),
            payoffs: report_average_mixed_reward.clone(),
        });
    }
    if !report_average_all_reward.is_empty(){
        payoff_series.push(PayoffGroupSeries{
            id: "All".to_string(),
            payoffs: report_average_all_reward.clone(),
        });
    }



    
    let payoff_plot_data_learning = PlotSeries {
        data: report_average_learning_reward,
        description: "Learning agents".to_string(),
        color: colors::BLACK,
    };

    let payoff_plot_data_all = PlotSeries {
        data: report_average_all_reward,
        description: "All agents".to_string(),
        color: colors::full_palette::GREY_A700,
    };

    let payoff_plot_data_hawk = PlotSeries {
        data: report_average_hawk_reward,
        description: "Hawk agents".to_string(),
        color: colors::RED,
    };

    let payoff_plot_data_dove = PlotSeries {
        data: report_average_dove_reward,
        description: "Dove agents".to_string(),
        color: colors::BLUE,
    };

    let payoff_plot_data_mixed = PlotSeries {
        data: report_average_mixed_reward,
        description: "Mixed agents".to_string(),
        color: colors::GREEN,
    };

    let mut plot_action_series = vec![];

    let plot_series_defect = PlotSeries {
        data: report_average_defects,
        description: "Defects".to_string(),
        color: colors::RED,
    };
    let plot_series_coops = PlotSeries {
        data: report_average_coops,
        description: "Cooperations".to_string(),
        color: colors::BLUE,
    };



    let mut plot_payoff_series = vec![];
    if !payoff_plot_data_learning.data.is_empty(){
        plot_payoff_series.push(payoff_plot_data_learning);
    }
    if !payoff_plot_data_hawk.data.is_empty(){
        plot_payoff_series.push(payoff_plot_data_hawk);
    }
    if !payoff_plot_data_dove.data.is_empty(){
        plot_payoff_series.push(payoff_plot_data_dove);
    }
    if !payoff_plot_data_all.data.is_empty(){
        plot_payoff_series.push(payoff_plot_data_all);
    }
    if !payoff_plot_data_mixed.data.is_empty(){
        plot_payoff_series.push(payoff_plot_data_mixed);
    }




    if !plot_series_defect.data.is_empty(){
        plot_action_series.push(plot_series_defect);
    }
    if !plot_series_coops.data.is_empty(){
        plot_action_series.push(plot_series_coops);
    }




    let stamp = chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]");
    let base_path = "results/replicator_dynamics_old/";
    std::fs::create_dir_all(base_path).unwrap();

    plot_many_series(Path::new(
        format!("{}/payoffs-replicator-{:?}_{}-{}-{}-{}_{}.svg",
                base_path,
                args.number_of_rounds,
                args.number_of_learning,
                args.number_of_hawks,
                args.number_of_doves,
                args.number_of_mixes,
                stamp
        ).as_str()), "",&plot_payoff_series[..],
        "Epoch",
        "Payoff"
    ).unwrap();

    plot_many_series(Path::new(
        format!("{}/learning_actions-replicator-{:?}_{}-{}-{}-{}_{}.svg",
                base_path,
                args.number_of_rounds,
                args.number_of_learning,
                args.number_of_hawks,
                args.number_of_doves,
                args.number_of_mixes,
                stamp
        ).as_str()), "",&plot_action_series[..],
        "Epoch",
        "Actions taken"
    ).unwrap();



    let file = File::create(
        format!("{}/payoffs-replicator-{:?}_{}-{}-{}-{}_{}.json",
                base_path,
                args.number_of_rounds,
                args.number_of_learning,
                args.number_of_hawks,
                args.number_of_doves,
                args.number_of_mixes,
                stamp).as_str()).unwrap();
    serde_json::to_writer(file, &payoff_series).unwrap();

    let file = File::create(
        format!("{}/game-trajectory-{:?}_{}-{}-{}-{}_{}.json",
                base_path,
                args.number_of_rounds,
                args.number_of_learning,
                args.number_of_hawks,
                args.number_of_doves,
                args.number_of_mixes,
                stamp).as_str()).unwrap();
    serde_json::to_writer_pretty(file, &model.environment.trajectory()).unwrap();
    


    Ok(())

}

