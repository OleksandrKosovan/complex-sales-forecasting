import pandas as pd
from statsforecast import StatsForecast
from fugue import transform
from datasetsforecast.losses import (
    mse, 
    mae,
    rmse,
    mape,
    smape,
)

from modeling.constants import (
    HORIZON, N_WINDOWS, FULL_RESULTS_PATH,
    EVALUATION_RESULTS_PATH, METRICS_RESULTS_PATH
)

import os
import logging
from time import time


class ClassicModel:
    def __init__(self, df: pd.DataFrame, models: list) -> None:
        self.df = df
        self.models = models
        self.metrics = [mse, mae, rmse, mape, smape]
    
    def modeling(self) -> pd.DataFrame:
        logging.info(f"Initialization of StatsForecast for {self.models}")
        sf = StatsForecast(
            models=self.models, 
            freq='D', 
            n_jobs=-1,
        )

        logging.info("Starting cross validation...")

        init = time()
        result = sf.cross_validation(
            df=self.df, 
            h=HORIZON, 
            n_windows=N_WINDOWS, 
            step_size=HORIZON, 
            level=[90]
        )
        end = time()

        logging.info(f"Forecast Minutes: {(end - init) / 60}")

        return result
    
    def evaludate_cross_validation(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        cv_results["unique_id"] = cv_results.index
        cv_results = cv_results.reset_index(drop=True)

        str_models = cv_results.loc[:, ~cv_results.columns.str.contains('unique_id|y|ds|cutoff|lo|hi')].columns
        str_models = ','.join([f"{model}:float" for model in str_models])
        
        evaluation_df = transform(
            cv_results.loc[:, ~cv_results.columns.str.contains('lo|hi')], 
            self.evaluate, 
            params={'metrics': self.metrics}, 
            schema=f"unique_id:str,cutoff:str,metric:str, {str_models}", 
            as_local=True,
            partition={'by': ['unique_id', 'cutoff']}
        )
        return evaluation_df

    def run(self):
        logging.info("Starting...")
        file_name = convert_classes_to_string(self.models) + ".csv"
        check_folders()

        result = self.modeling()

        logging.info(f"Saving CV results as {file_name}...")
        results_path = os.path.join(FULL_RESULTS_PATH, file_name, index=False)
        result.to_csv(results_path, index=False)

        logging.info("Evaluation...")
        evaluation = self.evaludate_cross_validation(result)

        logging.info(f"Saving evaluation as {file_name}...")
        evaluation.to_csv(os.path.join(EVALUATION_RESULTS_PATH, file_name), index=False)

        logging.info("Saving metrics...")
        metrics = evaluation.groupby(['metric']).mean(numeric_only=True)
        metrics.to_csv(os.path.join(METRICS_RESULTS_PATH, file_name), index=False)

        logging.info("Done!")

    @staticmethod
    def evaluate(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
        eval_ = {}
        models = df.loc[:, ~df.columns.str.contains('unique_id|y|ds|cutoff|lo|hi')].columns
        for model in models:
            eval_[model] = {}
            for metric in metrics:
                eval_[model][metric.__name__] = metric(df['y'], df[model])
        eval_df = pd.DataFrame(eval_).rename_axis('metric').reset_index()
        eval_df.insert(0, 'cutoff', df['cutoff'].iloc[0])
        eval_df.insert(0, 'unique_id', df['unique_id'].iloc[0])
        return eval_df


def convert_classes_to_string(class_list):
    class_string = '-'.join([obj.__class__.__name__ for obj in class_list])
    return class_string


def check_folders():
    folders = [
        FULL_RESULTS_PATH,
        EVALUATION_RESULTS_PATH, 
        METRICS_RESULTS_PATH
    ]
    for path in folders:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)


