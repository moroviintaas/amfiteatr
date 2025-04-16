
import argparse

from c4.a2c import PolicyA2C
from c4.agent import Agent
from c4.model import my_env, TwoPlayerModel
from pettingzoo.utils import env_logger




def parse_args():
    parser = argparse.ArgumentParser(description = "Baseline A2C model for ConnectFour Game")
    parser.add_argument('-e', "--epochs", type=int, default=100, help="Number of epochs of training")
    parser.add_argument('-g', "--games", type=int, default=128, help="Number of games in epochs of training")
    parser.add_argument('-s', "--steps", type=int,  dest="max_env_steps_in_epoch", help="Limit number of steps in train epoch (train epoch may be limited to certain number of steps to compare with models scaled with number of steps instead of full games)")
    parser.add_argument('-t', "--test_games", type=int, default=100, help="Number of games in epochs of testing")
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

    return parser.parse_args()




def main():
    env_logger.EnvLogger.suppress_output()

    args = parse_args()
    #print(args.layers1)
    #print(args.layers2)

    if args.cuda:
        dev = "cuda:0"
        device = dev
    else:
        dev = "cpu"
        device = dev


    env = my_env(render_mode=None)
    env.reset()
    policy0 = PolicyA2C(84, 7, args.layer_sizes_0, config=args, device=device )
    policy1 = PolicyA2C(84, 7, args.layer_sizes_1, config=args, device=device )
    agent0 = Agent("player_0", policy0)
    agent1 = Agent("player_1", policy1)

    model = TwoPlayerModel(env, agent0, agent1, args)

    #model.play_single_game(True, -3)

    w,c = model.play_epoch(args.test_games, False, args.penalty)
    print(f"Test after before training: wins: {w}, cheats: {c}")
    for e in range(args.epochs):
        w,c = model.play_epoch(args.games, True, args.penalty)
        print(f"Results train after epoch {e}: wins: {w}, cheats: {c}")
        ttrain_report = model.apply_experience()
        print(ttrain_report)
        w,c = model.play_epoch(args.test_games, False, args.penalty)
        print(f"Test after epoch {e}: wins: {w}, cheats: {c}")



if __name__ == "__main__":
    main()