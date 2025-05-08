
import argparse

from c4.a2c import PolicyA2C
from c4.agent import Agent
from c4.model import my_env, TwoPlayerModel
from pettingzoo.utils import env_logger

from c4.common import PolicyConfig
from c4.ppo import PolicyPPO
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description = "Baseline PPO model for ConnectFour Game")
    parser.add_argument('-e', "--epochs", type=int, default=100, help="Number of epochs of training")
    parser.add_argument('-E', "--extended-epochs", type=int, default=100, help="Number of extended epochs of training (only agent0 trains)")
    parser.add_argument('-g', "--games", type=int, default=128, help="Number of games in epochs of training")
    #parser.add_argument('-s', "--steps", type=int,  dest="max_env_steps_in_epoch", help="Limit number of steps in train epoch (train epoch may be limited to certain number of steps to compare with models scaled with number of steps instead of full games)")
    parser.add_argument("--limit-steps-in-epoch", type=int, default=None, help = "Limit number of all steps in all epoch")
    parser.add_argument('-t', "--test-games", type=int, default=100, help="Number of games in epochs of testing")
    parser.add_argument('-p', "--penalty", type=float, default=-10, help="NPenalty for illegal actions")
    parser.add_argument("--layer-sizes-0", metavar="LAYERS", type=int, nargs="*", default=[64,64], help = "Sizes of subsequent linear layers")
    parser.add_argument("--layer-sizes-1", metavar="LAYERS", type=int, nargs="*", default=[64,64],
                        help="Sizes of subsequent linear layers")
    parser.add_argument("--save-train-params-summary", dest = "save_path_train_param_summary", help = "File to save learn policy summary for epochs")
    parser.add_argument("--save-test-epoch-summary", dest = "save_path_test_epoch", help = "File to save test epoch average results")
    parser.add_argument("--save-train-epoch-summary", dest = "save_path_train_epoch", help = "File to save train epoch average results")
    parser.add_argument("--learning-rate", dest="learning_rate", default = 1e-4, help = "Adam learning rate")
    parser.add_argument("--masking", action="store_true", dest="masking",  help = "Enable illegal action masking")

    parser.add_argument("--cuda",  action="store_true", help="enable cuda")
    parser.add_argument("--gamma", dest="gamma", default = 0.99, help = "Discount factor gamma")
    parser.add_argument("--value-coefficient", dest="vf_coef", default = 0.5, help = "Value loss coeficient")
    parser.add_argument("--entropy-coefficient", dest="entropy_coef", type=float, default = 0.01, help = "Entropyloss coeficient")

    parser.add_argument('-m', "--minibatch-size", type=int, default=64, help="Size of PPO minibatch")
    parser.add_argument("-u", "--update-epochs", default=4, type=int, help="Nuber of update epochs inside PPO Policy")
    parser.add_argument("-G", "--gae-lambda", default=0.95, type=float, help="Lambda for GAE calculation (Advantage)")
    parser.add_argument("--clip-coefficient", default=0.2, help="Clipping coefficient for PPO")

    parser.add_argument("--tensorboard", help="Directory to write summary for model")
    parser.add_argument("--tensorboard-policy-agent-0", help="Directory to policy summary for agent 0")
    parser.add_argument("--tensorboard-policy-agent-1", help="Directory to policy summary for agent 1")

    return parser.parse_args()




def main():
    env_logger.EnvLogger.suppress_output()

    args = parse_args()

    policy_config = PolicyConfig(entropy_coef=args.entropy_coef, vf_coef=args.vf_coef, gamma = args.gamma,
                                 learning_rate=args.learning_rate, minibatch_size=args.minibatch_size,
                                 update_epochs=args.update_epochs, gae_lambda=args.gae_lambda,
                                 clip_coef=args.clip_coefficient)

    if args.tensorboard is not None:
        #tb_writter = None
        tb_writer = SummaryWriter(f"{args.tensorboard}", )
    else:
        tb_writer = None

    if args.tensorboard_policy_agent_0 is not None:
        #tb_writter = None
        tb_writer_policy_0 = SummaryWriter(f"{args.tensorboard_policy_agent_0}", )
    else:
        tb_writer_policy_0 = None

    if args.tensorboard_policy_agent_1 is not None:
        #tb_writter = None
        tb_writer_policy_1 = SummaryWriter(f"{args.tensorboard_policy_agent_1}", )
    else:
        tb_writer_policy_1 = None

    if args.tensorboard is not None:
        #tb_writter = None
        tb_writer = SummaryWriter(f"{args.tensorboard_policy_agent_0}", )
    else:
        tb_writer = None
#
    # if args.tensorboard_agent_1 is not None:
    #     #tb_writter = None
    #     tb_writer_1 = SummaryWriter(f"{args.tensorboard_policy_agent_1}", )
    # else:
    #     tb_writer_1 = None

    #policy_config.entropy_coef = arg
    #print(args.layers1)
    #print(args.layers2)

    if args.cuda:
        dev = "cuda:0"
        device = dev
    else:
        dev = "cpu"
        device = dev

    #agent_ids = ("player_0", "player_1")

    env = my_env(render_mode=None, illegal_reward=args.penalty)
    env.reset()
    policy0 = PolicyPPO(84, 7, args.layer_sizes_0, config=policy_config, device=device, tb_writer=tb_writer_policy_0, masking=args.masking )
    policy1 = PolicyPPO(84, 7, args.layer_sizes_1, config=policy_config, device=device, tb_writer=tb_writer_policy_1, masking=args.masking )
    agent0 = Agent("player_0", policy0,)
    agent1 = Agent("player_1", policy1)

    model = TwoPlayerModel(env, agent0, agent1, args)

    #model.play_single_game(True, -3)
    if args.test_games > 0:
        w,c, games, total_steps = model.play_epoch(args.test_games, False, args.penalty, args.limit_steps_in_epoch)
        print(f"Test after before training: wins: {w}, illegal: {c}")
    for e in range(args.epochs):
        w,c, games, total_steps = model.play_epoch(args.games, True, args.penalty, args.limit_steps_in_epoch)
        print(f"Results train after epoch {e}: wins: {w}, illegal: {c}")
        for agent_id in model.agents_ids:
            if model.agents[agent_id].policy.tb_writer is not None:
                #policy_id = model.agents[agent_id].policy.policy_id
                model.agents[agent_id].policy.tb_writer.add_scalar(f"train_epoch/score", w[agent_id] - w[model.other_agent(agent_id)], e)
                model.agents[agent_id].policy.tb_writer.add_scalar(f"train_epoch/illegal_moves", c[agent_id], e)
        train_report = model.apply_experience()
        print(train_report)
        if tb_writer is not None:
            tb_writer.add_scalar(f"train_epoch/number_of_games", games, e)
            tb_writer.add_scalar(f"train_epoch/number_of_steps_in_game", total_steps/games, e)
        if args.test_games > 0:
            w,c, games, total_steps = model.play_epoch(args.test_games, False, args.penalty)
            print(f"Test after epoch {e}: wins: {w}, illegal: {c}")

    for e in range(args.extended_epochs):
        w,c, games, total_steps = model.play_epoch(args.games, True, args.penalty, args.limit_steps_in_epoch)
        print(f"Results train after extended epoch {e}: wins: {w}, illegal: {c}")
        for agent_id in model.agents_ids:
            if model.agents[agent_id].policy.tb_writer is not None:
                #policy_id = model.agents[agent_id].policy.policy_id
                model.agents[agent_id].policy.tb_writer.add_scalar(f"train_epoch/score", w[agent_id] - w[model.other_agent(agent_id)], args.epochs + e)
                model.agents[agent_id].policy.tb_writer.add_scalar(f"train_epoch/illegal_moves", c[agent_id], args.epochs + e)
        #if tb_writer_0 is not None:
        #    #policy_id = model.agents[agent_id].policy.policy_id
        #    model.agents[agent_id].policy.tb_writer.add_scalar(f"train_epoch/wins/{agent_id}", w[agent_id], args.epochs + e)
        #    model.agents[agent_id].policy.tb_writer.add_scalar(f"train_epoch/illegal_moves/{agent_id}", c[agent_id], args.epochs + e)
        #train_report = model.apply_experience()
        agent0 = model.agents[model.agents_ids[0]]
        report_0 = model.agents[model.agents_ids[0]].policy.train_network(agent0.batch_trajectories)
        print(report_0)
        if tb_writer is not None:
            tb_writer.add_scalar(f"train_epoch/number_of_games", games, e + args.epochs)
            tb_writer.add_scalar(f"train_epoch/number_of_steps_in_game", total_steps/games, e + args.epochs)
        if args.test_games > 0:
            w,c, games, total_steps = model.play_epoch(args.test_games, False, args.penalty, args.limit_steps_in_epoch)
            print(f"Test after epoch {e}: wins: {w}, illegal: {c}")



if __name__ == "__main__":
    main()