mod options;

use std::{thread};
use std::path::{Path};
use log::{debug, info};
use amfiteatr_rl::tch::{Device, nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, VarStore};
use amfiteatr_rl::tensor_data::{TensorEncoding};
use amfiteatr_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorActorCritic};
use clap::{Parser};
use plotters::style::colors;
use amfiteatr_core::agent::*;
use amfiteatr_core::comm::EnvironmentMpscPort;
use amfiteatr_core::env::{AutoEnvironmentWithScores, ReseedEnvironment, ScoreEnvironment, TracingBasicEnvironment, TracingEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_classic::agent::{LocalHistoryInfoSet, LocalHistoryConversionToTensor, AgentAssessmentClassic};
use amfiteatr_classic::scheme::{AgentNum, ClassicScheme, ClassicGameSchemeNumbered};
use amfiteatr_classic::scheme::ClassicAction::Down;
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::SymmetricRewardTableInt;
use amfiteatr_rl::policy::*;
use crate::options::EducatorOptions;
use crate::options::SecondPolicy;
use amfiteatr_examples::plots::{plot_many_series, PlotSeries};
use amfiteatr_examples::series::{MultiAgentPayoffSeries, PayoffSeries};



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
type ClassicSchemeNumeric = ClassicScheme<AgentNum>;



pub fn run_game(
    env: &mut (impl AutoEnvironmentWithScores<ClassicSchemeNumeric> + Send + ReseedEnvironment<ClassicSchemeNumeric, ()>),
    agent0: &mut (impl Send + MultiEpisodeAutoAgent<ClassicSchemeNumeric, ()>),
    agent1: &mut (impl Send + MultiEpisodeAutoAgent<ClassicSchemeNumeric, ()>))
    -> Result<(), AmfiteatrError<ClassicSchemeNumeric>>{

    thread::scope(|s|{
        s.spawn(||{
            env.reseed(()).unwrap();
            env.run_with_scores().unwrap();
        });
        s.spawn(||{
            agent0.run_episode(()).unwrap()
        });
        s.spawn(||{
            agent1.run_episode(()).unwrap()
        });
    });
    Ok(())

}

type D = ClassicGameSchemeNumbered;


fn main() -> Result<(), AmfiteatrError<ClassicScheme<AgentNum>>>{

    let args = EducatorOptions::parse();
    setup_logger(&args).unwrap();
    let device = Device::Cpu;
    let number_of_players = 2;


    let reward_f: Box<dyn Fn(AgentAssessmentClassic<i64>) -> f32> = match args.policy{
        SecondPolicy::Std => Box::new(|reward| reward.table_payoff() as f32),
        SecondPolicy::MinDefects => {Box::new(|reward| reward.coops_as_reward() as f32)}
        SecondPolicy::StdMinDefects => Box::new(|reward|
            reward.f_combine_table_with_other_coop(args.reward_bias_scale * args.number_of_rounds as f32)),
        SecondPolicy::StdMinDefectsBoth => Box::new(|reward|{
            reward.f_combine_table_with_both_coop(args.reward_bias_scale * args.number_of_rounds as f32)
        }),
        SecondPolicy::Edu => Box::new(|reward|{
            reward.combine_edu_assessment(args.reward_bias_scale )
        }),
    };

    let tensor_repr = LocalHistoryConversionToTensor::new(args.number_of_rounds);

    let input_size = tensor_repr.desired_shape().iter().product();

    let mut payoffs_0 = Vec::with_capacity(args.epochs + 1);
    let mut payoffs_1 = Vec::with_capacity(args.epochs + 1);
    let mut custom_payoffs_1 = Vec::with_capacity(args.epochs + 1);
    let mut agent_0_coops = Vec::with_capacity(args.epochs + 1);
    let mut agent_1_coops = Vec::with_capacity(args.epochs + 1);
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
            .add_fn(|xs| xs.tanh())
            .add(nn::linear(path / "hidden2", 512, 256, Default::default()))
            .add_fn(|xs| xs.tanh())
            .add_fn(|xs|xs.relu());
            //.add(nn::linear(path / "h2", 512, 512, Default::default()));
        let actor = nn::linear(path / "al", 256, 2, Default::default());
        let critic =  nn::linear(path / "ac", 256, 1, Default::default());
        {move |input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
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
    //let test_policy = ClassicPureStrategy::new(ClassicAction::Defect);

    let net1 = A2CNet::new(VarStore::new(device), net_template.get_net_closure());
    let opt1 = net1.build_optimizer(Adam::default(), 1e-4).unwrap();
    let policy1 = ActorCriticPolicy::new(net1, opt1, tensor_repr, TrainConfig {gamma: 0.99});
    //let mut agent_1 = AgentGenT::new(state1, comm1, Arc::new(Mutex::new(policy1)));
    let mut agent_1 = TracingAgentGen::new(state1, comm1, policy1);


    //evaluate on start
    let mut coops = [Vec::new(), Vec::new()];
    let mut scores = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..100{
        debug!("Plaing round: {i:} of initial simulation");
        run_game(&mut environment, &mut agent_0, &mut agent_1)?;
        scores[0].push(agent_0.current_universal_score()) ;
        scores[1].push(agent_1.current_universal_score());
        //scores[2].push(reward_f(agent_1.current_assessment_total()) as i64);
        scores[2].push(reward_f(agent_1.info_set().current_assessment()) as i64);
        coops[0].push(agent_0.info_set().count_actions_self_calculate(Down));
        coops[1].push(agent_1.info_set().count_actions_self_calculate(Down));


    }
    let avg = [scores[0].iter().sum::<i64>() as f32/(scores[0].len() as f32),
            scores[1].iter().sum::<i64>() as f32/(scores[1].len() as f32),
            scores[2].iter().sum::<i64>() as f32/(scores[2].len() as f32)];
        info!("Average scores: 0: {}\t1:{}", avg[0], avg[1]);

    let coops_a = [coops[0].iter().map(|n| *n as i64).sum::<i64>() as f64 /(coops[0].len() as f64),
            coops[1].iter().map(|n| *n as i64).sum::<i64>() as f64/(coops[1].len() as f64),
    ];
    agent_0_coops.push(coops_a[0] as f32);
    agent_1_coops.push(coops_a[1] as f32);

    payoffs_0.push(avg[0]);
    payoffs_1.push(avg[1]);
    custom_payoffs_1.push(avg[2]);


    for e in 0..args.epochs{
        agent_0.clear_episodes()?;
        agent_1.clear_episodes()?;
        info!("Starting epoch {e:}");
        for _g in 0..args.batch_size{
            run_game(&mut environment, &mut agent_0, &mut agent_1)?;
        }
        let trajectories_0 = agent_0.take_episodes();
        let trajectories_1 = agent_1.take_episodes();
        agent_0.policy_mut().train(&trajectories_0[..])?;
        match args.policy{
            SecondPolicy::Std => agent_1.policy_mut().train(&trajectories_1[..]),
            SecondPolicy::MinDefects => {
                agent_1.policy_mut().train_generic(&trajectories_1[..], |step| {
                    //let own_defects = step.step_info_set().count_actions_self(Defect) as i64;
                    //let custom_reward = step.step_subjective_reward().count_other_actions(Cooperate);
                    //let custom_reward = reward_f(step.step_subjective_reward());
                    let custom_reward = reward_f(step.late_information_set().current_assessment() - step.information_set().current_assessment());
                    let v_custom_reward = [custom_reward];
                    //trace!("Calculating custom reward on info set: {}, with agent reward: {:?}.",
                    //    step.step_info_set(), step.step_subjective_reward());
                    //trace!("Custom reward calculated: {}", &custom_reward);
                    Tensor::from_slice(&v_custom_reward[..])
                })
            },

            SecondPolicy::StdMinDefects => {
                agent_1.policy_mut().train_generic(&trajectories_1[..], |step| {
                    //let own_defects = step.step_info_set().count_actions_self(Defect) as i64;
                    //let custom_reward = step.step_subjective_reward().f_combine_table_with_other_coop(100.0);
                    let custom_reward = reward_f(step.late_information_set().current_assessment() - step.information_set().current_assessment());
                    let v_custom_reward = [custom_reward];
                    //trace!("Calculating custom reward on info set: {}, with agent reward: {:?}.",
                    //    step.step_info_set(), step.step_subjective_reward());
                    //trace!("Custom reward calculated: {}", &custom_reward);
                    Tensor::from_slice(&v_custom_reward[..])
                })
            },

            SecondPolicy::StdMinDefectsBoth => {
                agent_1.policy_mut().train_generic(&trajectories_1[..], |step| {
                    let custom_reward = reward_f(step.late_information_set().current_assessment() - step.information_set().current_assessment());
                    let v_custom_reward = [custom_reward];

                    Tensor::from_slice(&v_custom_reward[..])
                })
            },
            SecondPolicy::Edu => {
                agent_1.policy_mut().train_generic(&trajectories_1[..], |step| {
                    let custom_reward = reward_f(step.late_information_set().current_assessment() - step.information_set().current_assessment());
                    let v_custom_reward = [custom_reward];

                    Tensor::from_slice(&v_custom_reward[..])
                })
            }
        }?;


        let mut scores = [Vec::new(), Vec::new(), Vec::new()];
        let mut coops = [Vec::new(), Vec::new()];
        for i in 0..100{
            debug!("Plaing round: {i:} of initial simulation");
            run_game(&mut environment, &mut agent_0, &mut agent_1)?;
            scores[0].push(agent_0.current_universal_score());
            scores[1].push(agent_1.current_universal_score());
            //scores[2].push(reward_f(agent_1.current_assessment_total()) as i64);
            scores[2].push(reward_f(agent_1.info_set().current_assessment()) as i64);
            coops[0].push(agent_0.info_set().count_actions_self_calculate(Down));
            coops[1].push(agent_1.info_set().count_actions_self_calculate(Down));

        }

        let avg = [scores[0].iter().sum::<i64>() as f32 /(scores[0].len() as f32),
            scores[1].iter().sum::<i64>() as f32/(scores[1].len() as f32),
            scores[2].iter().sum::<i64>() as f32/(scores[2].len() as f32),

        ];
        let coops_a = [coops[0].iter().map(|n| *n as i64).sum::<i64>() as f64 /(coops[0].len() as f64),
            coops[1].iter().map(|n| *n as i64).sum::<i64>() as f64/(coops[1].len() as f64),
        ];
        debug!("Score sums: {scores:?}, of size: ({}, {}).", scores[0].len(), scores[1].len());
        info!("Average scores: 0: {}\t1: {}", avg[0], avg[1]);
        payoffs_0.push(avg[0]);
        payoffs_1.push(avg[1]);
        custom_payoffs_1.push(avg[2]);
        agent_0_coops.push(coops_a[0] as f32);
        agent_1_coops.push(coops_a[1] as f32);
    }

    run_game(&mut environment, &mut agent_0, &mut agent_1)?;
    println!("{:?}", agent_0.take_episodes().last().unwrap().last_view_step().unwrap());



    println!("{}", environment.trajectory().last_view_step().unwrap());

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

    let agent1_custom_data = PlotSeries {
        data: custom_payoffs_1,
        description: "Agent 1 - self assessment".to_string(),
        color: colors::GREEN,
    };

    let agent1_coops = PlotSeries {
        data: agent_1_coops,
        description: "Agent 1 cooperations".to_string(),
        color: colors::BLUE,
    };
    let agent0_coops = PlotSeries {
        data: agent_0_coops,
        description: "Agent 0 cooperations".to_string(),
        color: colors::RED
    };

    let s_policy = match args.policy{
        SecondPolicy::StdMinDefects => {
            format!("{:?}-{:?}", SecondPolicy::StdMinDefects, args.reward_bias_scale)
        },
        SecondPolicy::Edu => "edu".to_string(),
        a => format!("{:?}", a)
    };
    let stamp = chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]");
    let base_path = "results/custom_assessment/";

    let mut series = MultiAgentPayoffSeries::<D>{
        agent_series: vec![],
    };
    series.agent_series.push(PayoffSeries{
        id: *agent_0.id(),
        payoffs: agent0_data.data.clone()

    });
    series.agent_series.push(PayoffSeries{
        id: *agent_1.id(),
        payoffs: agent1_data.data.clone()

    });

    let plot_series = match args.policy{
        SecondPolicy::Std => vec![agent0_data, agent1_data],
        _ => vec![agent0_data, agent1_data, agent1_custom_data]
    };

    plot_many_series(Path::new(
        format!("{}/payoffs-{}-{:?}_{}.svg",
                base_path,
                &s_policy.as_str(),
                args.number_of_rounds,
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"))
            .as_str(), ),  "",&plot_series[..],
        "Epoch",
        "Payoff"
    ).unwrap();

    plot_many_series(Path::new(
        format!("{}/actions-1l-{}-{:?}_{}.svg",
                base_path,
                &s_policy.as_str(),
                args.number_of_rounds,
                stamp)
            .as_str(), ), "",&[agent0_coops, agent1_coops,],
            "Epoch",
            "Cooperations"
            ).unwrap();
    //plot_payoffs(Path::new(format!("custom-payoffs-{:?}-{:?}.svg", args.policy, args.number_of_rounds).as_str()), &agent1_custom_data ).unwrap();

    Ok(())
    //let standard_strategy =
}