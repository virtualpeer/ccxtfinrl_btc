from BtcTradingEnv3 import BtcTradingEnv
import gym
from elegantrl.agent import AgentDQN  # AgentDQN
from elegantrl.run import *
import yfinance as yf
from stockstats import StockDataFrame as Sdf
import torch

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