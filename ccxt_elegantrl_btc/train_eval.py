from BtcTradingEnv3 import BtcTradingEnv
import gym
from elegantrl.tutorial.env import PreprocessEnv, get_gym_env_info
from elegantrl.run import *
import yfinance as yf
from stockstats import StockDataFrame as Sdf


'''choose an DRL algorithm'''
from elegantrl.agent import AgentDQN  # AgentDQN

args = Arguments(agent=None, env=None, gpu_id=0)
args.agent = AgentDQN()

'''choose environment'''
args.env = BtcTradingEnv(if_train=True)
args.env_eval = BtcTradingEnv(if_train=False)
args.rollout_num = 2
args.gamma = 0.995
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8
"TotalStep: 2e3, TargetReward: , UsedTime: 10s"

# args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
# args.net_dim = 2 ** 8
# args.batch_size = 2 ** 8

'''train and evaluate'''
train_and_evaluate(args)

args = Arguments(agent=None, env=None, gpu_id=0)
args.agent = AgentDQN()
args.env = BtcTradingEnv(if_train=False)
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8
args.if_remove = False
args.cwd = './AgentDQN/BtcTradingEnv-v3_0'
args.init_before_training()
# Draw the graph
BtcTradingEnv(if_train=False)\
    .draw_cumulative_return(self = args.env, _torch = torch)
