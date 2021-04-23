# ccx_elegantrl_btc

First run train_eval.py file. It trains and evaluates the model. The cumulative return and the model are saved under './AgentDQN/BtcTradingEnv-v3_0'

The target reward is set to be 1.04 in the BtcTradingEnv3.py file. It is often the case that the target reward cannot converges to such a level. (So it keeps training.) You may KeyboardInterrupt the training process by pressing ctrl+C, then run load_eval.py file to load the saved model and evaluate it.

You can see some results under ./results .  The subfile is named by explore_rate for agentDQN. For example, explore_rate_01 means the explore_rate is set to be 0.1 for agentDQN. The change of explore_rate is achieved by changing explore_rate in elegantrl/agent.py. (The target_reward in BtcTradingEnv-v3 is also set individually when producing these results)

# notes
The 'btc_spot_future1.csv' file is fetched by ccxt and then preprocessed by finrl.preprocessing.preprocessors.FeatureEngineer.
See https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/notebooks/Crypto_Binance_Historical_Data.ipynb, and
https://github.com/AI4Finance-LLC/FinRL/blob/master/finrl/preprocessing/preprocessors.py.

The codes for preprocessing are not yet uploaded.

