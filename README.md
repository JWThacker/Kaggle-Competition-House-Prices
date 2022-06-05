# Kaggle Competition: Hous Prices - Advanced Regression Techniques

## Link to Medium Article
https://medium.com/@thacker.jared/my-first-data-science-competition-9b856538ea3d

## Skills Used
  * Exploratory data analysis and visualization
  * Data manipulation and cleaning
  * Feature engineering and selection
  * Model selection and evaluation

## Libraries Used
There were several Python libraries that I used for this competition, a few notable ones are:
  * Pandas (data manipulation/cleaning)
  * Seaborn (data visualization)
  * Scikit-Learn (Machine learning models)
  * XGBoost(Extreme Gradient Boosted Trees)

## Project Description/Motivation
From Kaggle:
"Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home."

In a nutshell the goal of this project is to predict the sale price of a property based on the features, or different representations of the features, provided. I evaluated several different modes for this task but settled on a stacking regressor that used a ridge regression and XGBoost regressor for the stacking layers and a linear regression model for the meta-learner.

## File Descriptions
  * **eda.ipynb**

    This is a Python notebook that performs the exploratory data analysis necessary before I begin the modeling process.

  * **modeling.ipynb**

    This is a Python notebook that performs the modeling selection and evaluation. After these actions are performed, the predictions are made on the test data and submitted to the competition.

  * **house-prices-advanced-regressoin-techniques**

    This is a directory containing the data that was provided by Kaggle.com.

  * **pipelines**

    A Python module containing the machine-learning pipelines for all of the different models I cross-validate. I packaged these pipelines so that the modeling.ipynb notebook did not seem so cumbersome.

  * **visuals**

    A Python module containing functions that modularize common visualizations that I like to make during EDA.

  * **submission.csv**

    A CSV containing my submission predictions to the Kaggle.com competition.

  * **test.csv**

    This is the test data provided by Kaggle.com *but* it has all of the engineered features that I created in eda.ipynb.

  * **train.csv**

    This is the train data provided by Kaggle.com *but* it has all of the engineered features that I created in eda.ipynb.

  * **requirements.txt**

    A flat file containing all of the Python packages and modules necessary to run the python notebooks.

## Results

  Exploratory Data Analysis: There were several variables that had the overwhelming majority of their values missing - these were dropped. The other missing values were due to a house not having a garage or a basement or both.

  Feature Engineering: Most of the time spent here was on binning the imbalanced categorical features (factors) so that one class/level did not dominated the training process - in the future I would like to conduct a correspondence analysis on the factors/categorical features. I used PCA to reduce the number of numerical features. If I could go back, I would spend more time on engineering the numerical features and *would not* perform PCA (probably).

  Model Evaluation and Tuning: Grid search combined with 5-fold cross-validation was used to evaluate and select the best performing hyperparameters from all of the models.

  Model Selection: The models were compared using the *negative-mean-squared-log-error** - this was per the rules of the competition. This metric was visualized with a point plot that contained the RMSLE for all of the models. It was found that *KNN* had the worst performance of all 5 models. *ridge regression*, *linear regression*, and *XGBoost* all had similar performance. The best performing model was a stacking regressor that was constructed by stacking the *ridge regression* and  *XGBoost* models with the highest performing parameters from the **Model Evaluation** step. The final reported validation **RMSLE was 0.014404** and the final reported test **RMSLE was 0.15356** - this is a clear case of overfitting since the test error was about an order of magnitude greater than the training error. I believe this is due to the presence of the XGBoost model in the second stacking layer. Nevertheless, the aforementioned testing error was enough to get me ranked 2531 out of 4382 competitors which I am pleased with as a first attempt at this problem.

  Future Ideas For Improving the Model: I think one of the keys to this competition is feature engineering, and specifically engineering the numerical features which I kind of punted on in order to focus on the categorical features. I have *a lot* of ideas for potential feature engineering that I could perform in order to increase the accuracy of the model. In addition, I believe including more models in the stacking layers of the *stacking regressor* will greatly improve the accuracy of the model. Specifically, I think adding lower-variance (but higher bias) models (like KNN with a relatively higher K value) with the addition of variables that can capture the non-linearity (polynomial terms, interaction terms etc) in the response will achieve a lower RMSLE.

## How to Interact With This Repository
```bash
git clone https://github.com/JWThacker/Kaggle-Competition-House-Prices.git
cd ./Housing_Prices_Project
```
## Installations
```bash
pip install -r requirements.txt
```
## Acknowledgements
All of the work you see here are my own ideas - I used nobody else's code. The goal of this project and every project I do on Kaggle is to learn and not to place highly - if I place highly, awesome, otherwise I'm very happy with just learning and gaining experience.

The only acknowledgement I would like to give is to *Applied Predictive Modeling* by Max Kuhn et al. (2013) from which I am learning how to be a competent modeler and of course the instruction from Udacity.

## License (MIT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
