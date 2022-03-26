import json
import logging
import deepchecks
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from deepchecks import Dataset
from deepchecks.base import suite
from deepchecks.suites import single_dataset_integrity


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

        if len(self.raw_config.get('ProcessingInputs', [])) != 1:
            raise ValueError(f"Expected data input, got "
                             f"{len(self.raw_config.get('ProcessingInputs', []))}")

        self.input_path = Path(self.raw_config[
            'ProcessingInputs'][0][
            'S3Input'][
            'LocalPath'].rstrip('/')).resolve()

        print(f'input path: {self.input_path}')


def adjust_series_for_daylight_saving_time(s: pd.Series, freq='H') -> pd.Series:
    new_series = s.copy()
    new_series = new_series.loc[~s.index.duplicated(keep='first')]
    new_index = pd.date_range(s.index.min(), s.index.max(), freq=freq, name=s.index.name)
    new_series = new_series.reindex(new_index)
    new_series.interpolate()
    return new_series


def main():
    logger = logging.getLogger()

    # Load config

    config = ProcessingJobConfig()

    interim_data_path = config.output_path / "interim"
    interim_data_path.mkdir(parents=True, exist_ok=True)
    processed_data_path = config.output_path / "processed"
    processed_data_path.mkdir(parents=True, exist_ok=True)

    check_results_path = config.output_path / "check_results"
    check_results_path.mkdir(parents=True, exist_ok=True)


    # Load data

    logger.info('Processing entsoe-2006...')
    enstoe_2006_raw_df = pd.read_excel(config.input_path / 'entsoe-2006' / '2006_2015_data.xlsx', engine='openpyxl')
    enstoe_2006_df = enstoe_2006_raw_df.drop(columns=['Country', 'Coverage ratio'])
    enstoe_2006_df = pd.melt(enstoe_2006_df, id_vars=['Year', 'Month', 'Day'], var_name='Hour', value_name='Load')
    enstoe_2006_df['Date'] = pd.to_datetime(enstoe_2006_df.drop(columns='Load'))
    enstoe_2006_df = enstoe_2006_df.set_index('Date').drop(columns=['Year', 'Month', 'Day', 'Hour'])

    logger.info(f'After melting, found {enstoe_2006_df.Load.isna().sum()} NaNs. Interpolating...')
    enstoe_2006_df['Load'] = adjust_series_for_daylight_saving_time(enstoe_2006_df['Load'])

    logger.info(f'Processed entsoe-2006. Saving...')
    enstoe_2006_interim_path = interim_data_path / '2006_2015_data.parquet'
    enstoe_2006_df.to_parquet(enstoe_2006_interim_path)
    logger.info(f'Saved entsoe-2006 to {enstoe_2006_interim_path}.')

    logger.info('Processing entsoe-2016...')
    enstoe_2016_raw_df = pd.read_excel(config.input_path / 'entsoe-2016' / '2016_2017_data.xlsx', engine='openpyxl')
    enstoe_2016_df = enstoe_2016_raw_df.loc[:, ['DateUTC', 'Value_ScaleTo100']]
    enstoe_2016_df = enstoe_2016_df.rename(columns={'Value_ScaleTo100': 'Load'})

    logger.info('Converting timezones...')
    enstoe_2016_df['DateUTC'] = enstoe_2016_df['DateUTC'].dt.tz_localize('UTC')
    enstoe_2016_df['Date'] = enstoe_2016_df['DateUTC'].dt.tz_convert('Europe/Rome').dt.tz_localize(None)
    enstoe_2016_df = enstoe_2016_df.drop(columns='DateUTC').set_index('Date')
    enstoe_2016_df = enstoe_2016_df[:'2016']

    logger.info(f'After conversion, dataframe has shape {enstoe_2016_df.shape}')
    enstoe_2016_df['Load'] = adjust_series_for_daylight_saving_time(enstoe_2016_df['Load'])
    logger.info(f'After re-indexing and interpolation, dataframe has shape {enstoe_2016_df.shape}')

    logger.info(f'Processed entsoe-2016. Saving...')
    enstoe_2016_interim_path = interim_data_path / '2016_data.parquet'
    enstoe_2016_df.to_parquet(enstoe_2016_interim_path)
    logger.info(f'Saved entsoe-2016 to {enstoe_2016_interim_path}.')

    logger.info('Processing Terna files...')
    file_list = list((config.input_path / 'terna').rglob("????_data.xlsx"))
    logger.info(f"{len(file_list)} raw input data files found: {[f.resolve().name for f in file_list]}")

    logger.info("Reading terna input data files...")
    terna_df = pd.concat([pd.read_excel(f, engine='openpyxl') for f in file_list])

    logger.info('Resampling Terna data...')
    terna_df['Date'] = pd.to_datetime(terna_df['Date'])
    terna_df = terna_df.drop(columns=['Forecast Total load [MW]']).rename(columns={'Total Load [MW]': 'Load'})
    terna_df = terna_df.groupby(['Date']).agg({'Load': 'sum'})
    terna_df = terna_df.resample('H').mean()
    terna_df['Load'] = adjust_series_for_daylight_saving_time(terna_df['Load'])
    logger.info(f'After resampling, data has shape {terna_df.shape}')

    logger.info(f'Processed terna. Saving...')
    terna_interim_path = interim_data_path / '2017_2022_data.parquet'
    terna_df.to_parquet(terna_interim_path)
    logger.info(f'Saved terna to {terna_interim_path}.')

    logger.info('Concatenating data...')
    interim_paths = list(interim_data_path.rglob('*.parquet'))

    logger.info(f'Found {len(interim_paths)} interim files: {[f.resolve().name for f in interim_paths]}')
    processed_df = pd.concat([pd.read_parquet(f) for f in interim_paths])
    cleaned_df = processed_df.sort_index().replace(0, np.nan).interpolate()

    logger.info(f'Saving processed dataset')
    processed_path = processed_data_path / '2006_2022_data.parquet'
    cleaned_df.to_parquet(processed_path)
    logger.info(f'Saved processed dataset to {processed_path}.')

    #deepchecks
    suite = single_dataset_integrity()

    data_path_faulty = config.input_path / "ingestion" / "faulty_ingestion_data.xlsx"
    raw_ds_faulty = Dataset(
        pd.read_excel(data_path_faulty, index_col=0, parse_dates=True, na_filter=False), cat_features=["Bidding zone"]
    )
    result_faulty = suite.run(raw_ds_faulty)
    result_faulty.save_as_html(check_results_path / "faulty_ingestion_data.html")

    data_path_good = config.input_path / "ingestion" / "good_ingestion_data.xlsx"
    raw_ds_good = Dataset(
        pd.read_excel(data_path_good, index_col=0, parse_dates=True, na_filter=False), cat_features=["Bidding zone"]
    )
    result_good = suite.run(raw_ds_good)
    result_good.save_as_html(check_results_path / "good_ingestion_data.html")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    print(f'pandas version: {pd.__version__}')
    main()
