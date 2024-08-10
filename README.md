# 207 Final Project - Zillow

A repository to collaborate on the 207 final project with the Zillow dataset from Kaggle. 

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

## Model 4: NN + XGBoost

## How to run the pipeline/installation

## Contribution Table



[keras-url]: https://keras.io/
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
