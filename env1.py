import numpy as np
import gym
import pandas as pd
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import datetime

class BtcTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset, initial_account=1e6, transaction_fee_percent=1e-3, if_train=True,
                 if_short_selling=False, max_step = None): #dataset: pd dataframe
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.df = dataset
        self.if_train = if_train
        self.if_short_selling = if_short_selling
        self.start_time = dataset[0:1]['date'].values[0]
        self.time = 0
        self.initial_account__reset = self.initial_account
        self.account = self.initial_account__reset
        self.current_df = self.df[self.time:self.time+1]
        self.total_asset = self.account 
        self.position = 0
        self.episode = 0
        self.episode_return = 0.0  
        self.gamma_return = 0.0
        self.reward = 0.0
        self.asset_memory = [self.account]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        '''env information'''
        self.env_name = 'BtcTradingEnv-v1'
        self.data_list = ['open', 'high', 'close', 'volume', 'macd',
                              'boll_ub','boll_lb','rsi_30', 'cci_30','dx_30',
                              'close_30_sma','close_60_sma', 'turbulence']
        self.state_dim = len(['account','total_asset','position']\
                             + self.data_list)
        self.action_space = spaces.Box(low = -1, high = 1,shape = (1,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_dim,))
        account_info = [self.account] + [self.total_asset] \
         + [self.position] 
        indicators = sum([self.current_df[ind].values.tolist() for ind in self.data_list], [])
        self.state = np.array(account_info + indicators)
        self.action_dim = 1
        self.if_discrete = False
        self.target_reward = 1.25  # convergence 1.5
        if max_step == None:
            self.max_step = self.df.shape[0]
        else:
            self.max_step = min(max_step, self.df.shape[0])

    def reset(self):
        self.initial_account__reset = self.initial_account  # reset()
        self.account = self.initial_account__reset
        self.total_asset = self.account
        self.time = 0
        self.current_df = self.df[self.time:self.time+1]
        self.episode_return = 0.0
        self.position = 0
        self.reward = 0.0
        account_info = [self.account] + [self.total_asset] \
         + [self.position] 
        indicators = sum([self.current_df[ind].values.tolist() for ind in self.data_list], [])
        self.state = np.array(account_info + indicators)
        self.episode = self.episode + 1       
        return self.state

    def step(self, action):
        close = self.current_df['close'].values[0]
        next_total_asset = self.account + self.position*close
        self.reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * 0.99 + self.reward  # notice: gamma_r seems good? Yes
        available_amount = (self.account // (close/1e8))/1e8 
        if action > 0:  # buy_stock
            delta_stock = min(action*10, available_amount)
            self.account -= close * delta_stock * (1 + self.transaction_fee_percent)
            self.position += delta_stock
        elif action < 0:  # sell_stock
            if self.if_short_selling == False:
                delta_stock = min(-action*10, self.position)
                self.account += close * delta_stock * (1 - self.transaction_fee_percent)
                self.position -= delta_stock
            elif self.if_short_selling == True:
                delta_stock = min(-action*10, self.position + \
                                  0.5*(self.total_asset // (close/1e8))/1e8)
                self.account += close * delta_stock * (1 - self.transaction_fee_percent)
                self.position -= delta_stock
                    

        """update day"""
        self.current_df = self.df[self.time :self.time+1]
        self.time += 1
        done = self.time == self.max_step  
        account_info = [self.account] + [self.total_asset] \
        + [self.position] 
        indicators = sum([self.current_df[ind].values.tolist() for ind in self.data_list], [])
        self.state = np.array((account_info + indicators), dtype = object)
        self.rewards_memory.append(self.reward)
        self.asset_memory.append(self.total_asset)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(action)

        if done:
            self.reward += self.gamma_return
            self.gamma_return = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = self.total_asset / self.initial_account
            print(self.episode_return)

        return self.state, self.reward, done, {}
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        action_list = self.actions_memory
        df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _get_date(self):
        timestamp = self.start_time//1000 + self.time*60
        date = datetime.datetime.fromtimestamp(timestamp)
        return date

processed = pd.read_csv('processed_t.csv', index_col=0)
env = BtcTradingEnv(dataset=processed)
from stable_baselines3.common.env_checker import check_env
check_env(env)




