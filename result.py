import pandas as pd
import os

pwd = os.path.abspath(__file__)
base_dir = os.path.dirname(pwd)

result_dir = os.path.join(base_dir, 'Train4Paper')

result_list = os.listdir(result_dir)

for model_name in result_list :
    result_path = os.path.join(result_dir, model_name, 'train_result.csv')
    if os.path.isfile(result_path) :


        df = pd.read_csv(result_path)

        idx_min = df['val_loss'].argmin()

        result_val = df['val_valence'].loc[idx_min]
        result_aro = df['val_arousal'].loc[idx_min]

        print('{:<22} [Valence : {:.2f}, Arousal : {:.2f}, Epoch : {}]'.format(model_name, result_val, result_aro, df['iteration'].loc[idx_min]//320739))

    else :
        print('{:<22} is not tested yet.'.format(model_name))
