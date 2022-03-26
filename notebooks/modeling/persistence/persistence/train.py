#!/usr/bin/env python

import json
import os
import sys
import traceback
import warnings

import joblib
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

warnings.filterwarnings("ignore")

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

channel_name = 'training'
training_path = os.path.join(input_path, channel_name)


def train():
    print('Start training script')
    try:

        # Read in any param that the user passed with the training job
        with open(param_path, 'r') as tc:
            training_params = json.load(tc)
        print(f"training_params: {training_params}")

        # Load Data
        input_files = [os.path.join(training_path, file) for file in os.listdir(training_path)]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))
        raw_data = [pd.read_parquet(file) for file in input_files]
        raw_df = pd.concat(raw_data).asfreq('D')
        load_train = raw_df['Load']

        naive_forecaster = NaiveForecaster(
            strategy=training_params['strategy'],
            sp=int(training_params['sp'])
        )
        naive_forecaster.fit(load_train)

        model_filename = os.path.join(model_path, 'model.joblib')
        joblib.dump(naive_forecaster, model_filename)
        print(f'Model has been saved in {model_filename}')
        print("\n\nTraining complete!")
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
