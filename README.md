# Numerai Tournament Machine Learning Model

This is a Python script that builds a machine learning model to predict the target variable in a Numerai tournament dataset. The script performs feature engineering, hyperparameter tuning, ensembling, regularization, validation, handling missing values, dimensionality reduction, and model selection. The script also uses CatBoostRegressor and LGBMRegressor models to predict the target variable and ensemble the predictions. The performance of the models is evaluated using 5-fold cross-validation and spearman correlation coefficient.

## Getting Started

### Prerequisites

The following libraries are required to run the script:

* numerapi
* catboost
* scipy
* numpy
* pandas
* sklearn
* lightgbm

### Installing

To install the required libraries, run the following command:

```
!pip install numerapi catboost scipy numpy pandas matplotlib sklearn lightgbm
```
### Usage

To use the script, simply run it in a Python environment, such as Jupyter Notebook. The script downloads the dataset automatically and saves the predictions in a submission file.

### License

This project is licensed under the MIT License - see the LICENSE file for details.
