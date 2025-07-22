import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('../../data/clean/data_ND300_dupl_rem.csv')

print('Data loaded')

rf = RandomForestRegressor(
    n_estimators=933,
    max_features='sqrt',
    n_jobs=-1
)

rf.fit(data.iloc[:, :-1], data.iloc[:, -1])

print('Model trained... now saving it')

out_file = ('../../all_result/rf/rf_ND300_final.model')

with open(out_file, 'wb') as file:
    pickle.dump(obj=rf,file=file,  protocol=pickle.HIGHEST_PROTOCOL)

print('Done!')
