import pandas as pd
from finrl.config import config
from finrl.preprocessing.preprocessors import FeatureEngineer

# local csv to df
df = pd.read_csv('btc_usdt_1m.csv', names=['date','open','high',
                                           'low','close','volume'])
# insert cryptocurrency type
df.insert(loc=1, column='tic',value='btc')


fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)
        
processed = fe.preprocess_data(df)
processed.to_csv('processed.csv')
