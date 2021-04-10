import env_btc
import pandas as pd
from finrl.preprocessing.data import data_split
from finrl.model.models import DRLAgent

start_time = '1609429500000'
mid = str(int(start_time)+1000*60*30721)
end_time = str(int(start_time)+1000*60*50001)
processed = pd.read_csv('processed.csv', index_col=0)
train = data_split(processed, int(start_time), int(mid))
trade = data_split(processed, int(mid), int(end_time))  
env_train = env_btc.BtcTradingEnv(dataset=train, if_short_selling=False)
agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=30720)
env_trade = env_btc.BtcTradingEnv(dataset=trade, if_short_selling = False)
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ppo, 
    environment = env_trade)

# df_account_value.to_csv('account_value.csv')
print('return:' + str(((df_account_value[-2:-1]['account_value'].values[0])/1e6)-1))

