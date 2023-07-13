import os
import logging
import pandas as pd

from modeling.constants import DATA_FOLDER, M5_FILE, HORIZON
from modeling.data_preprocessing import preprocess_m5
from modeling.train import ClassicModel

from statsforecast.models import (
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
    WindowAverage,
    SeasonalWindowAverage,
)



logging.info("Start reading data...")
df = pd.read_parquet(os.path.join(DATA_FOLDER, M5_FILE))
logging.info("Finished reading data.")

models = [
    HistoricAverage(),
    Naive(),
    RandomWalkWithDrift(),
    SeasonalNaive(season_length=HORIZON),
    WindowAverage(window_size=HORIZON),
    SeasonalWindowAverage(season_length=HORIZON, window_size=HORIZON),
]

if __name__ == '__main__':
    inst = ClassicModel(df, models)
    inst.run()
