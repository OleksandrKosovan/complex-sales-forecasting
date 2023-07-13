import os
import pandas as pd

from modeling.constants import DATA_FOLDER, M5_FILE


def preprocess_m5():
    df = pd.read_parquet('https://m5-benchmarks.s3.amazonaws.com/data/train/target.parquet')
    df = df.rename(columns={
        'item_id': 'unique_id', 
        'timestamp': 'ds', 
        'demand': 'y'
    })
    df['ds'] = pd.to_datetime(df['ds'])
    df['unique_id'] = df['unique_id'].astype(str)
    df = df.reset_index(drop=True)
    df.to_parquet(os.path.join(DATA_FOLDER, M5_FILE), compression='gzip') 
    print("The file is saved.")


def preprocess_fozzy():
    # todo: implement function
    pass
