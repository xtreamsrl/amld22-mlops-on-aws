import sys
import logging
import argparse
import joblib
import os
import subprocess
import json

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# In order to use the feature store, we need to update boto3
subprocess.check_call([sys.executable, "-m", "pip", "install", '--upgrade', 'boto3'])

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info(f'boto3 version: {boto3.__version__}')

session = boto3.Session()
featurestore_client = session.client(service_name='sagemaker-featurestore-runtime', region_name='eu-west-1')

MODEL_NAME = 'model.joblib'


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--log_transform_target', type=int, default=1)

    args = parser.parse_args()

    # Read train and test data
    train_df = pd.read_parquet(os.path.join(args.train, 'train.parquet'))
    test_df = pd.read_parquet(os.path.join(args.test, 'test.parquet'))

    # Split train and test data
    train_x, train_y = train_df.drop(columns='Load'), train_df['Load']
    test_x, test_y = test_df.drop(columns='Load'), test_df['Load']

    # Use scikit-learn's LinearRegression to train the model.
    if args.log_transform_target:
        model = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)
    else:
        model = LinearRegression()
    model.fit(train_x, train_y)
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)

    # Log metrics to make them appear in the experiment registry
    logger.info(f'Train MAPE: {mean_absolute_percentage_error(train_y.values, train_predictions)};')
    logger.info(f'Train MAE: {mean_absolute_error(train_y.values, train_predictions)};')
    logger.info(f'Train RMSE: {np.sqrt(mean_squared_error(train_y.values, train_predictions))};')

    logger.info(f'Test MAPE: {mean_absolute_percentage_error(test_y.values, test_predictions)};')
    logger.info(f'Test MAE: {mean_absolute_error(test_y.values, test_predictions)};')
    logger.info(f'Test RMSE: {np.sqrt(mean_squared_error(test_y.values, test_predictions))};')

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(model, os.path.join(args.model_dir, MODEL_NAME))


def model_fn(model_dir):
    """
    Deserialize and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load(os.path.join(model_dir, MODEL_NAME))
    return model


def input_fn(request_body, request_content_type):
    """
    Preprocess input data from incoming requests. Must return a suitable format for the serialized model
    """
    request_data = json.loads(request_body)
    featurestore_response = featurestore_client.batch_get_record(
        Identifiers=[{"FeatureGroupName": 'load-forecasting', "RecordIdentifiersValueAsString": request_data['dates']}]
    )

    records = []
    feature_dates = []  # dates are not returned in the same order as in the request
    for item in featurestore_response['Records']:
        feature_dict = {f['FeatureName']: float(f['ValueAsString']) for f in item['Record'] if
                        f['FeatureName'] not in {'date', 'event_time'}}
        records.append(feature_dict)
        feature_dates.append(next(f['ValueAsString'] for f in item['Record'] if f['FeatureName'] == 'date'))

    input_df = pd.DataFrame.from_records(records, index=pd.to_datetime(feature_dates)).sort_index()
    return input_df
