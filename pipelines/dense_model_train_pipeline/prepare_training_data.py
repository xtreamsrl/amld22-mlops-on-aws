import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _install_with_pip(pkg_name: str) -> None:
    subprocess.call([sys.executable, "-m", "pip", "install", pkg_name])


def _update_with_pip(pkg_name: str) -> None:
    subprocess.call([sys.executable, "-m", "pip", "install", pkg_name, "--upgrade"])


def build_lagged_df(series: pd.Series, n_lags: int) -> pd.DataFrame:
    df = pd.DataFrame({series.name: series})
    for i in range(1, n_lags + 1):
        df[f'{series.name}_{i}'] = series.shift(i)
    df = df.dropna()
    return df


class ProcessingJobConfig:
    def __init__(self):
        self.config_path = '/opt/ml/config/processingjobconfig.json'
        try:
            with open(self.config_path, 'r') as f:
                self.raw_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'{self.config_path} not found')
        except IOError:
            raise ValueError(f"No job configuration file found at {self.config_path}")
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Processing job configuration file cannot be parsed as json")

        if len(self.raw_config.get('ProcessingOutputConfig', {}).get('Outputs', [])) != 1:
            raise ValueError(f"Expected one output, got "
                             f"{len(self.raw_config.get('ProcessingOutputConfig', {}).get('Outputs', []))}")

        print(self.raw_config)

        self.output_path = Path(self.raw_config[
            'ProcessingOutputConfig'][
            'Outputs'][0][
            'S3Output'][
            'LocalPath'].rstrip('/')).resolve()

        print(f'out path: {self.output_path}')

        if len(self.raw_config.get('ProcessingInputs', [])) != 2:
            raise ValueError(f"Expected code and data input, got "
                             f"{len(self.raw_config.get('ProcessingInputs', []))}")

        self.input_path = Path(self.raw_config[
            'ProcessingInputs'][0][
            'S3Input'][
            'LocalPath'].rstrip('/')).resolve()

        print(f'input path: {self.input_path}')


def main():
    # Load config
    config = ProcessingJobConfig()

    out_path = config.output_path / 'train_data'
    out_path.mkdir(parents=True, exist_ok=True)

    # Load data
    now = '2019-12-31 23:59'
    raw_df = pd.read_parquet(config.input_path / '2006_2022_data.parquet')
    
    logger.info("Raw data read from S3. Proceed with processing...")
    complete_data_df = raw_df.resample("D").sum()
    data_df = complete_data_df[:now]
    load_series = data_df['Load']

    covid_df = complete_data_df[:'2020-05-31'].copy()
    covid_len = covid_df.shape[0] - data_df.shape[0]

    logger.info("Generating lagged features...")
    n_lags = 7
    covid_max = covid_df[:now].Load.max()
    ff_df = build_lagged_df(covid_df.Load / covid_max, n_lags=n_lags)
    x_train_ff_scaled, x_test_ff_scaled, y_train_ff_scaled, y_test_ff_scaled = train_test_split(
        ff_df.drop(columns=['Load']),
        ff_df.Load,
        test_size=covid_len,
        shuffle=False
    )
    logger.info("Processing completed. Saving data...")

    train_df = ff_df.copy().loc[x_train_ff_scaled.index]
    train_df.to_parquet(out_path / 'train.parquet', engine='fastparquet')
    logger.info(f"Data saved in {out_path / 'train.parquet'}")


if __name__ == '__main__':
    print('starting')
    _install_with_pip(pkg_name='fastparquet')
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    print(f'pandas version: {pd.__version__}')
    main()
