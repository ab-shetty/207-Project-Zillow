# 207 Final Project - Zillow

A repository to collaborate on the 207 final project with the Zillow dataset from Kaggle. 

## Table of Contents
- [Built With](#built-with)
- [Introduction: Zillow Prize: Zillow’s Home Value Prediction (Zestimate)](#introduction-zillow-prize-zillows-home-value-prediction-zestimate)
- [Model 1: Linear Regression](#model-1-linear-regression)
- [Model 2: Neural Network](#model-2-neural-network)
- [Model 3: XGBoost](#model-3-xgboost)
- [Model 4: NN + XGBoost](#model-4-nn--xgboost)
- [How to Run the Pipeline/Installation](#how-to-run-the-pipelineinstallation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Choose Your Environment](#choose-your-environment)
  - [Run the Pipeline](#run-the-pipeline)
  - [Example Commands](#example-commands)
  - [Understanding the Script](#understanding-the-script)
  - [Generating the Submission File](#generating-the-submission-file)
  - [Notes](#notes)
  - [Troubleshooting](#troubleshooting)
- [Contribution Table](#contribution-table)


### Built With

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
* ![Optuna](https://img.shields.io/badge/Optuna-8A2BE2)
* ![XGBoost](https://img.shields.io/badge/XGBoost-8A2BE2)


## Introduction: Zillow Prize: Zillow’s Home Value Prediction (Zestimate)
https://www.kaggle.com/competitions/zillow-prize-1

**Question:** Can we improve the Zillow Zestimate? YES
Specifically, predict the log-error between Zillow’s Zestimate and the actual sale price, given all the features of a home. The log error is defined as
𝑙𝑜𝑔𝑒𝑟𝑟𝑜𝑟=𝑙𝑜𝑔(𝑍𝑒𝑠𝑡𝑖𝑚𝑎𝑡𝑒)−𝑙𝑜𝑔(𝑆𝑎𝑙𝑒𝑃𝑟𝑖𝑐𝑒)

**Motivation (Why is it interesting?):** 
* Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning
* $1M Kaggle competition closed in 2018
* Well defined problem
* Scores from the Kaggle leaderboard are published to benchmark our models

**Data:** Full list of real estate properties in three counties (Los Angeles, Orange and Ventura, California) data in 2016 for train and validation. Zillow maintains public (2016) and private (2017) data for testing used for competition scoring.

**Modeling/Experiments:**
* Multi linear regression - Worse than Zillow
* Neural Networks - Better than Zillow
* Gradient Boosting Algorithms - Better than Zillow
* Combination (Neural Network with XGBoost) - Better than Zillow

**Future Work:** Feature Engineering, Explore other Combination Models


## Model 1: Linear Regression

We decided to start with trying a multi-linear regression (MLR) algorithm as it’s simple, interpretable, and useful for understanding the impact of each feature on the target variable. However, we expect the multi-linear regression (MLR) model not to do well, as it is unable to capture non-linear relationships.

We started with looking at all of the features in the properties_2016 dataset. We narrowed the features of interest based on which features didn’t have too many missing values or that could reasonably have the missing values replaced or dropped. 
Next, we looked at which optimizer would perform best on the MLR model while holding all other hyperparameters equal. We looked at multiple optimizers, including Adadelta, Adafactor, Adagrad, Adam, Adamax, Lion, Nadam, and SGD. ‘Adam’ performed the best, so we chose that optimizer for our final MLR model.

We then tested various epochs, starting from 5 up to 30. We found 10 epochs were enough to reach a stabilized mean absolute loss. 

Finally, we experimented with a range of learning rates, starting from 0.0001 up to 0.001. We found that 0.007 helped reach a stable, low enough mean absolute loss. 

Upon submitting the model to Kaggle to compare with the Zillow baseline, we found that the MLR model overall performed worse compared to the baseline, which was within our expectations.

We concluded that the MLR model is likely inappropriate for this model, as lowering the MAE any further would likely require significantly more time investment, so we will evaluate other models first before considering MLR again as an optimal model if needed.

## Model 2: Neural Network

## Model 3: XGBoost
Gradient Boosting Algorithms: Scalable boosted tree algorithm, where trees are built in parallel. It minimizes the loss function by adding models sequentially, focusing on reducing the errors made by the previous models. Excellent for handling complex, non-linear relationships in the data, and tabular data.  XGBoost was released March 27, 2014, a few years before the Zillow competition and not widely used. It started gaining in popularity, and XGBoost was named among InfoWorld’s coveted Technology of the Year award winners in 2019. XGBoost is easy to implement. The run time is fast, and it provides excellent results. XGBoost performed well against the 2016 Zillow ZEstimate. It had a 0.00134 improvement for the private score over the Zillow baselise, and 0.00104 improvement for the public score.

The train vs validation MAE were within 1%, so it generalized well.

The Optuna library for hyperparameter tuning with the following parameter values:
* Lambda (1e-8 to 1, log=true) - L2 regularization term on weights
* Alpha (1e-8 to 1, log=true) - L1 regularization term on weights
* Eta (0.01 to 0.1) - learning rate step size the optimizer makes updates to the weights 
* Max_depth (1 to 9) - maximum depth of the tree models
* Subsample (0.6 to 1.0) - fraction of observations used for each tree
* Colsample_bytree (0.6 to 1.0) - fraction of features used for each tree
* N_estimators (100 to 1000) - number of trees in the model

Gradient Boosting Algorithms: Scalable boosted tree algorithm, where trees are built in parallel. It minimizes the loss function by adding models sequentially, focusing on reducing the errors made by the previous models. Excellent for handling complex, non-linear relationships in the data, and tabular data.  XGBoost was released March 27, 2014, a few years before the Zillow competition and not widely used. It started gaining in popularity, and XGBoost was named among InfoWorld’s coveted Technology of the Year award winners in 2019. XGBoost is easy to implement. The run time is fast, and it provides excellent results. XGBoost performed well against the 2016 Zillow ZEstimate. It had a 0.00134 improvement for the private score over the Zillow baselise, and 0.00104 improvement for the public score.

The train vs validation MAE were within 1%, so it generalized well.

The Optuna library for hyperparameter tuning with the following parameter values:
* Lambda (1e-8 to 1, log=true) - L2 regularization term on weights
* Alpha (1e-8 to 1, log=true) - L1 regularization term on weights
* Eta (0.01 to 0.1) - learning rate step size the optimizer makes updates to the weights 
* Max_depth (1 to 9) - maximum depth of the tree models
* Subsample (0.6 to 1.0) - fraction of observations used for each tree
* Colsample_bytree (0.6 to 1.0) - fraction of features used for each tree
* N_estimators (100 to 1000) - number of trees in the model

## Model 4: NN + XGBoost

## How to Run the Pipeline/Installation
This section provides a step-by-step guide on how to set up and run the Zillow project pipeline. You will learn how to clone the repository, install the necessary dependencies, and execute the Python script that allows you to train different models or use a pretrained model to generate predictions.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.7 or higher
- pip (Python package installer)

### Clone the Repository

Start by cloning the repository to your local machine using the following command:
```bash
git clone https://github.com/ab-shetty/207-Project-Zillow
```
Next, you can cd into the repository
```bash
cd zillow-project
```


### Install Dependencies 
Once you have cloned the repository, install the required dependencies using pip. You can find these dependencies listed in the requirements.txt file. Run the following command to install them:
```bash
pip install -r requirements.txt
```
### Choose Your Environment 
The script supports two environments: `local` and `colab`. You need to specify which environment you are working in when you run the script.
* `Local`: Use this option if you are running the script on your local machine and have the data stored in the  `data` folder within the project directory.
* `Colab`: Use this option if you are running the script on Google Colab. You will need to mount your Google Drive and provide the path to the data.

### Run the Pipeline
You can run the pipeline by executing the `main.py` script. This script provides several options depending on what you want to do:
* `Train a Model`: You can choose which model to train: Linear Regression, Neural Network, XGBoost, or a combination model.
* `Use a Pretrained Model for Predictions`: You can load a pretrained model to generate predictions, which will create a submission file for Kaggle.

### Example Commands
Train a Linear Regression Model Locally:

If you want to train a Linear Regression model on your local machine:
```bash
python main.py --env local --model linear_regression
```
Train a Neural Network Model on Google Colab:
If you are working on Google Colab and want to train a Neural Network model:

```bash
python main.py --env colab --model neural_network
```

Or, Generate Predictions Using a Pretrained Model:
To load a pretrained model and generate predictions (for example, on your local machine):
```bash
python main.py --env local --predict /path/to/pretrained/model
```

### 5. Understanding the Script

The `main.py` script is the entry point for running the Zillow project pipeline. Here’s a breakdown of what each option does:

- **Environment Setup (`--env`)**:

  This option lets you specify whether you are working locally or on Google Colab.

- **Model Selection (`--model`)**:

  Choose which model you want to train:
  
  - `linear_regression`: Trains a Linear Regression model.
  - `neural_network`: Trains a Neural Network model.
  - `xgboost`: Trains an XGBoost model.
  - `combination`: Trains a combination model (Neural Network + XGBoost).

- **Load Pretrained Model and Predict (`--predict`)**:

  Use this option if you want to load a pretrained model to generate predictions. You need to provide the path to the pretrained model file.

### 6. Generating the Submission File

After running the model, the script will automatically generate a submission file in CSV format, which you can use to submit predictions to Kaggle.

### 7. Notes

- Ensure that the data is correctly placed in the specified path (either locally or on Google Drive) before running the script.
- If you are using Google Colab, remember to mount your Google Drive before running the script to ensure it can access your data.

### 8. Troubleshooting

- **Missing Dependencies**: If you encounter errors related to missing dependencies, double-check that you have installed all the required packages listed in `requirements.txt`.
- **File Not Found**: Ensure that your data files are in the correct location as expected by the script. For local runs, they should be in the `data` folder; for Colab, you need to provide the correct Google Drive path.


<br>

## Contribution Table
<br>

| Name               | Section Worked On                        | Hours Spent |
|--------------------|------------------------------------------|-------------|
| Abhishek Shetty    | **Feature Engineering**                  | 8           |
|                    | - Creating New Features                  | 5           |
|                    | - Feature Selection                      | 3           |
|                    | **Combination Model (NN + XGBoost)**     | 12          |
|                    | - Model Development                      | 7           |
|                    | - Training and Integration               | 5           |
|                    | **Hyperparameter Tuning**                | 6           |
|                    | - Tuning Combined Model                  | 6           |
|                    | **Total**                                | **26**      |
| Ahmeda Cheick      | **Exploratory Data Analysis (EDA)**      | 10          |
|                    | - Data Cleaning                          | 4           |
|                    | - Visualizations                         | 3           |
|                    | **Neural Networks**                      | 12          |
|                    | - Model Architecture                     | 5           |
|                    | - Training and Evaluation                | 7           |
|                    | **Total**                                | **22**      |
| Athena Le          | **Multi-linear Regression**              | 11          |
|                    | - Model Development                      | 8           |
|                    | - Model Evaluation                       | 3           |
|                    | **Model Architecture**                   | 11           |
|                    | - Designing Model Structure              | 7           |
|                    | - Evaluating Performance                 | 4          |
|                    | **Total**                                | **22**      |
| Phillip Hoang      | **Feature Engineering**                  | 8           |
|                    | - Creating New Features                  | 5           |
|                    | - Feature Selection                      | 3           |
|                    | **XGBoost**                              | 10          |
|                    | - Model Development                      | 6           |
|                    | - Model Tuning                           | 4           |
|                    | **Model Architecture**                   | 6           |
|                    | - Designing Model Structure              | 4           |
|                    | - Hyperparameter Tuning                  | 2           |
|                    | **Total**                                | **24**      |


[keras-url]: https://keras.io/
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
