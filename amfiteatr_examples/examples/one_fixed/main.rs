mod options;

use std::{thread};
use std::fs::File;
use std::path::{Path};
use log::{debug, info};
use amfiteatr_rl::tch::{Device, nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, VarStore};
use amfiteatr_rl::tensor_data::{ConversionToTensor};
use amfiteatr_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorA2C};
use clap::{Parser};
use plotters::style::colors;
use amfiteatr_core::agent::*;
use amfiteatr_core::comm::EnvironmentMpscPort;
use amfiteatr_core::env::{AutoEnvironmentWithScores, ReseedEnvironment, ScoreEnvironment, TracingBasicEnvironment, TracingEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_classic::agent::{FibonacciForgiveStrategy, LocalHistoryInfoSet, LocalHistoryInfoSetNumbered, LocalHistoryConversionToTensor, SwitchAfterTwo};
use amfiteatr_classic::domain::{AgentNum, ClassicGameDomain, ClassicGameDomainNumbered};
use amfiteatr_classic::domain::ClassicAction::{Down, Up};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::policy::ClassicMixedStrategy;
use amfiteatr_classic::SymmetricRewardTableInt;
use amfiteatr_rl::policy::{ActorCriticPolicy, LearningNetworkPolicy, TrainConfig};
use crate::options::EducatorOptions;
use crate::options::SecondPolicy;
use amfiteatr_examples::plots::{plot_many_series, PlotSeries};
use amfiteatr_examples::series::{MultiAgentPayoffSeries, PayoffSeries};

/*
pub struct ModelElements<ID: UsizeAgentId, Seed>{
    pub environment: Arc<Mutex<dyn AutoEnvironmentWithScores<ClassicGameDomain<ID>>>>,
    agents: [Arc<Mutex<dyn AutomaticAgentRewarded<ClassicGameDomain<ID>>>>;2],
    seed: PhantomData<Seed>,
}

 */

pub fn setup_logger(options: &EducatorOptions) -> Result<(), fern::InitError> {
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
        .level(options.log_level)
        .level_for("amfiteatr_examples", options.log_level)
        .level_for("amfiteatr_core", options.log_level_amfi);

        match &options.log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        .apply()?;
    Ok(())
}
type Domain = ClassicGameDomain<AgentNum>;
//type A2C = ActorCriticPolicy<D, OwnHistoryInfoSetNumbered, OwnHistoryTensorRepr>;


pub fn run_game(
    env: &mut (impl AutoEnvironmentWithScores<Domain> + Send + ReseedEnvironment<Domain, ()>),
    agent0: &mut (impl AutomaticAgent<Domain> + Send + ReseedAgent<Domain, ()> + MultiEpisodeAutoAgent<Domain, ()>),
    //agent1: &mut (impl MultiEpisodeAgent<Domain, ()> + AutomaticAgentRewarded<Domain> + Send + ReseedAgent<Domain, ()>)
    agent1: &mut Box<dyn ModelAgent<Domain, (), LocalHistoryInfoSetNumbered>>
    )
    -> Result<(), AmfiteatrError<Domain>>{

    thread::scope(|s|{
        s.spawn(||{
            env.reseed(()).unwrap();
            env.run_with_scores().unwrap();
        });
        s.spawn(||{
            agent0.reseed(()).unwrap();
            agent0.run_episode(()).unwrap()
        });
        s.spawn(||{
            //let mut g = agent1.lock().unwrap();
            //g.run_episode_rewarded(()).unwrap()

            agent1.reseed(()).unwrap();
            agent1.run_episode(()).unwrap()
        });
    });
    Ok(())

}

pub enum AgentWrap{
    //Learning(Arc<Mutex<dyn NetworkLearningAgent<InfoSetType=(), Policy=()>>>),
    //Simple(Arc<Mutex<dyn Au>>)
}
type D = ClassicGameDomainNumbered;

fn main() -> Result<(), AmfiteatrError<ClassicGameDomain<AgentNum>>>{

    let args = EducatorOptions::parse();
    setup_logger(&args).unwrap();
    let device = Device::Cpu;
    //type Domain = ClassicGameDomainNumbered;
    let number_of_players = 2;



    let tensor_repr = LocalHistoryConversionToTensor::new(args.number_of_rounds);

    let input_size = tensor_repr.desired_shape().iter().product();

    let mut payoffs_0 = Vec::with_capacity(args.epochs + 1);
    let mut payoffs_1 = Vec::with_capacity(args.epochs + 1);
    let mut agent_1_coops = Vec::with_capacity(args.epochs + 1);
    let mut agent_1_defects = Vec::with_capacity(args.epochs + 1);
    //let mut custom_payoffs_1 = Vec::with_capacity(args.epochs + 1);
    //let mut opti_payoffs_1 = Vec::with_capacity(args.epochs + 1);



    let mut env_adapter = EnvironmentMpscPort::new();
    let comm0 = env_adapter.register_agent(0).unwrap();
    let comm1 = env_adapter.register_agent(1).unwrap();

    let reward_table = SymmetricRewardTableInt::new(
        args.coop_versus_coop,
        args.coop_versus_defect,
        args.defect_versus_coop,
        args.defect_versus_defect);


    let net_template = NeuralNetTemplate::new(|path|{
        let seq = nn::seq()
            .add(nn::linear(path / "input", input_size, 512, Default::default()))
            //.add(nn::linear(path / "h1", 256, 256, Default::default()))
            .add(nn::linear(path / "hidden1", 512, 512, Default::default()))
            .add(nn::linear(path / "hidden2", 512, 512, Default::default()))
            .add_fn(|xs|xs.relu());
            //.add(nn::linear(path / "h2", 512, 512, Default::default()));
        let actor = nn::linear(path / "al", 512, 2, Default::default());
        let critic =  nn::linear(path / "ac", 512, 1, Default::default());
        {move |input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });







    let env_state_template = PairingState::new_even(number_of_players, args.number_of_rounds, reward_table.into()).unwrap();
    let mut environment = TracingBasicEnvironment::new(env_state_template.clone(), env_adapter);


    let net0 = A2CNet::new(VarStore::new(device), net_template.get_net_closure());
    let opt0 = net0.build_optimizer(Adam::default(), 1e-4).unwrap();
    let normal_policy = ActorCriticPolicy::new(net0, opt0, tensor_repr, TrainConfig {gamma: 0.99});
    let state0 = LocalHistoryInfoSet::new(0, reward_table.into());
    let mut agent_0 = TracingAgentGen::new(state0, comm0, normal_policy);


    let state1 = LocalHistoryInfoSet::new(1, reward_table.into());

    let mut agent_1: Box<dyn ModelAgent<D, (), LocalHistoryInfoSetNumbered, >> = match args.policy{
        SecondPolicy::Mixed => {
            Box::new(TracingAgentGen::new(state1, comm1, ClassicMixedStrategy::new(args.defect_proba as f64)))
        }
        SecondPolicy::SwitchTwo => {Box::new(TracingAgentGen::new(state1, comm1, SwitchAfterTwo{}))}
        SecondPolicy::FibonacciForgive => {Box::new(TracingAgentGen::new(state1, comm1, FibonacciForgiveStrategy{}))},
        SecondPolicy::ForgiveAfterTwo => {Box::new(TracingAgentGen::new(state1, comm1, amfiteatr_classic::agent::ForgiveAfterTwo{}))}
    };


    //evaluate on start
    let mut scores = [Vec::new(), Vec::new()];
    let mut actions = [Vec::new(), Vec::new()];
    for i in 0..100{
        debug!("Plaing round: {i:} of initial simulation");
        //let mut agent_1_guard = agent_1.lock().unwrap();
        run_game(&mut environment, &mut agent_0, &mut agent_1)?;
        scores[0].push(agent_0.current_universal_score()) ;
        scores[1].push(agent_1.current_universal_score());
        actions[0].push(agent_0.info_set().count_actions_self_calculate(Down));
        actions[1].push(agent_0.info_set().count_actions_self_calculate(Up));
        //scores[2].push(reward_f(agent_1.current_subjective_score()) as i64);


    }
    let avg = [scores[0].iter().sum::<i64>()/(scores[0].len() as i64),
        scores[1].iter().sum::<i64>()/(scores[1].len() as i64),
            //scores[2].iter().sum::<i64>()/(scores[2].len() as i64)
    ];
    let avg_a = [actions[0].iter().map(|n| *n as i64).sum::<i64>() as f64 /(actions[0].len() as f64),
        actions[1].iter().map(|n| *n as i64).sum::<i64>() as f64/(actions[1].len() as f64),
    ];
        info!("Average scores: 0: {}\t1:{}", avg[0], avg[1]);

    payoffs_0.push(avg[0] as f32);
    payoffs_1.push(avg[1] as f32);
    agent_1_coops.push(avg_a[0] as f32);
    agent_1_defects.push(avg_a[1] as f32);
    //custom_payoffs_1.push(avg[2] as f32);


    for e in 0..args.epochs{
        agent_0.clear_episodes();
        agent_1.clear_episodes();
        info!("Starting epoch {e:}");
        for _g in 0..args.batch_size{
            run_game(&mut environment, &mut agent_0, &mut agent_1)?;
        }
        let trajectories_0 = agent_0.take_episodes();
        //let trajectories_1 = agent_1.take_episodes();
        agent_0.policy_mut().train_on_trajectories_env_reward(&trajectories_0[..])?;



        let mut scores = [Vec::new(), Vec::new(), Vec::new()];
        let mut actions = [Vec::new(), Vec::new()];
        for i in 0..100{
            debug!("Plaing round: {i:} of initial simulation");
            run_game(&mut environment, &mut agent_0, &mut agent_1)?;
            scores[0].push(agent_0.current_universal_score());
            scores[1].push(agent_1.current_universal_score());
            actions[0].push(agent_0.info_set().count_actions_self_calculate(Down));
            actions[1].push(agent_0.info_set().count_actions_self_calculate(Up));
            //scores[2].push(reward_f(agent_1.current_subjective_score()) as i64);

        }

        let avg = [scores[0].iter().sum::<i64>() as f64 /(scores[0].len() as f64),
            scores[1].iter().sum::<i64>() as f64/(scores[1].len() as f64),
        ];
        let avg_a = [actions[0].iter().map(|n| *n as i64).sum::<i64>() as f64 /(actions[0].len() as f64),
            actions[1].iter().map(|n| *n as i64).sum::<i64>() as f64/(actions[1].len() as f64),
        ];
        debug!("Score sums: {scores:?}, of size: ({}, {}).", scores[0].len(), scores[1].len());
        info!("Average scores: 0: {}\t1: {}", avg[0], avg[1]);
        payoffs_0.push(avg[0] as f32);
        payoffs_1.push(avg[1] as f32);
        agent_1_coops.push(avg_a[0] as f32);
        agent_1_defects.push(avg_a[1] as f32);
        //custom_payoffs_1.push(avg[2] as f32);
    }

    run_game(&mut environment, &mut agent_0, &mut agent_1)?;
    //println!("{:?}", agent_0.take_episodes().last().unwrap().list().last().unwrap());



    println!("{}", environment.trajectory().number_of_steps());

    println!("Scores: 0: {},\t1: {}", environment.actual_score_of_player(&0), environment.actual_score_of_player(&1));


    //plot_payoffs(Path::new(format!("agent_0-{:?}-{:?}.svg", args.policy, args.number_of_rounds).as_str()), &payoffs_0[..]).unwrap();
    //plot_payoffs(Path::new(format!("agent_1-{:?}-{:?}.svg", args.policy, args.number_of_rounds).as_str()), &payoffs_1[..]).unwrap();

    let agent0_data = PlotSeries {
        data: payoffs_0,
        description: "Agent 0".to_string(),
        color: colors::RED,
    };
    let agent1_data = PlotSeries {
        data: payoffs_1,
        description: "Agent 1".to_string(),
        color: colors::BLUE,
    };

    let agent1_coops = PlotSeries {
        data: agent_1_coops,
        description: "Agent 1 cooperations".to_string(),
        color: colors::BLUE,
    };
    let agent1_defects = PlotSeries {
        data: agent_1_defects,
        description: "Agent 1 defects".to_string(),
        color: colors::RED
    };




    let s_policy = match args.policy{
        SecondPolicy::Mixed => {format!("mixed-{:.02}", args.defect_proba)}
        SecondPolicy::SwitchTwo => {"switch2".to_string()}
        SecondPolicy::FibonacciForgive => {"fibonacci".to_string()},
        SecondPolicy::ForgiveAfterTwo => "forgive_2coops".to_string(),
    };
    let stamp = chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]");
    let base_path = "results/one_fixed/";
    std::fs::create_dir_all(base_path).unwrap();

    let mut series = MultiAgentPayoffSeries::<D>{
        agent_series: vec![],
    };
    series.agent_series.push(PayoffSeries{
        id: *agent_0.info_set().agent_id(),
        payoffs: agent0_data.data.clone()

    });
    series.agent_series.push(PayoffSeries{
        id: *agent_1.info_set().agent_id(),
        payoffs: agent1_data.data.clone()

    });

    plot_many_series(Path::new(
        format!("{}/payoffs-1l-{}-{:?}_{}.svg",
                base_path,
                &s_policy.as_str(),
                args.number_of_rounds,
                stamp)
            .as_str()), "",&[agent0_data, agent1_data,],
        "Epoch",
        "Payoff"
    ).unwrap();
    //plot_payoffs(Path::new(format!("custom-payoffs-{:?}-{:?}.svg", args.policy, args.number_of_rounds).as_str()), &agent1_custom_data ).unwrap();

    plot_many_series(Path::new(
        format!("{}/actions-1l-{}-{:?}_{}.svg",
                base_path,
                &s_policy.as_str(),
                args.number_of_rounds,
                stamp)
            .as_str(), ), "", &[agent1_coops, agent1_defects,],
        "Epoch",
        "Actions taken"
    ).unwrap();

    let file = File::create(
        format!("{}/payoffs-1l-{}-{:?}_{}.json",
                base_path,
                &s_policy.as_str(),
                args.number_of_rounds,
                stamp).as_str()).unwrap();
    serde_json::to_writer(file, &series).unwrap();

    if let Some(agent_0_trace) = agent_0.episodes().last(){
        let file_trace_0 = File::create(
            format!(
            "{}/trace0-1l-{}-{:?}-{}.json",
                base_path,
                &s_policy.as_str(),
                args.number_of_rounds,
                stamp).as_str()
        ).unwrap();

        serde_json::to_writer(file_trace_0, agent_0_trace).unwrap()
    }
    /*
    if let Some(agent_1_trace) = agent_1.episodes().last(){
        let file_trace_1 = File::create(
            format!(
            "{}/trace1-1l-{}-{:?}-{}.json",
                base_path,
                &s_policy.as_str(),
                args.number_of_rounds,
                stamp).as_str()
        ).unwrap();

        serde_json::to_writer(file_trace_1, agent_1_trace).unwrap()
    }

     */
    Ok(())
    //let standard_strategy =
}

