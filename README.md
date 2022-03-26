# MLOPS on AWS | AMLD 2022

This repository contains all the resources necessary to follow along and reproduce the workshop "*MLOps on AWS: a Hands-On Tutorial*", which will be held at the Applied Machine Learning Days (AMLD) 2022. You can find more information about the workshop [here](https://appliedmldays.org/events/amld-epfl-2022/workshops/mlops-on-aws-a-hands-on-tutorial).

## Folders structure
```
.
|
├───cdk                                      # CDK project to build an AWS VPC for the workshop
|
├───data
│   ├───processed                            # results of data processing and merging
│   └───raw                                  # data downloaded from the source
|
├───notebooks
│   ├───eda                                  # Exploratory Data Analysis
│   └───modeling
│       ├───deepar                           # Short-term forecasting with DeepAR (Algo mode)
│       ├───dense-network                    # Short-term forecasting with a Feedforward neural net (Framework mode)
│       ├───fourier-regression               # Long-term forecasting using Fourier regression (Framework mode)
│       └───persistence                      # Long-term forecasting using a containerized Naive classifier (BYOC mode)
|
├───pipelines
│   ├───data_preprocessing                   # Setup and run the data ingestion preprocessing pipeline example
│   └───dense_model_train_pipeline           # Resources to build a Sagemaker pipeline with a Tensorflow model
|
└───slides                                   # Introduction to MLOps and AWS
```

## Workshop overview
  
Applying machine learning in the real world is hard: reproducibility gets lost, datasets are dirty, data flows break down, the context where models operate keeps evolving. In the last 2-3 years, the emerging MLOps paradigm provided a strong push towards more structured and resilient workflows.

MLOps is about supporting and automating the assessment of model performance, model deployment and the following monitoring. Valuable tools for an effective MLOps process are data version trackers, model registries, feature stores, and experiment trackers.

During the workshop, we will showcase the challenges of “applied” machine learning and the value of MLOps with a practical case study. We will develop a ML model following MLOps best practices, from raw data to production deployment. Then, we will simulate a further iteration of development, resulting in better performance, and we will appreciate how MLOps allows for easy comparison and evolution of models.

AWS will provide the tools to effectively implement MLOps: the workshop is also intended to offer an overview of the main resources of the cloud platform and to show how they can support model development and operation.
