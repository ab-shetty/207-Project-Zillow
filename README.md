# 207 Final Project - Zillow

A repository to collaborate on the 207 final project with the Zillow dataset from Kaggle. 

## Table of Contents
- [Built With](#built-with)
- [Model 1: Linear Regression](#model-1-linear-regression)
- [Model 2: Neural Network](#model-2-neural-network)
- [Model 3: XGBoost](#model-3-xgboost)
- [Model 4: NN + XGBoost](#model-4-nn--xgboost)
- [How to Run the Pipeline/Installation](#how-to-run-the-pipelineinstallation)
- [Contribution Table](#contribution-table)

### Built With

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
* ![Optuna](https://img.shields.io/badge/Optuna-8A2BE2)
* ![XGBoost](https://img.shields.io/badge/XGBoost-8A2BE2)

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
. Lambda (1e-8 to 1, log=true) - L2 regularization term on weights
. Alpha (1e-8 to 1, log=true) - L1 regularization term on weights
. Eta (0.01 to 0.1) - learning rate step size the optimizer makes updates to the weights 
. Max_depth (1 to 9) - maximum depth of the tree models
. Subsample (0.6 to 1.0) - fraction of observations used for each tree
. Colsample_bytree (0.6 to 1.0) - fraction of features used for each tree
. N_estimators (100 to 1000) - number of trees in the model

## Model 4: NN + XGBoost

## How to Run the Pipeline/Installation

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
