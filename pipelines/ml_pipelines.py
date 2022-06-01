# Imports
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression

"""Apply treatment coding to categorical variables

Inherits: BaseEstimator, TransformerMixin
"""
class CustomEncoder(BaseEstimator, TransformerMixin):

    """Constructor for CustomEncoder
    
    Positional Arguments:
        feats: a list of categorical variables to encode
    """
    def __init__(self, feats):
        self.feats = feats
    
    """fit data to this transformer

    Positional Arguments:
        X - model matrix of predictors
    """
    def fit(self, X, y=None):
        return self
    
    """transform data using treatment coding

    Positional Arguments:
        X - model matrix of predictors
    """
    def transform(self, X, y=None):
        encoder = OneHotEncoder(drop='first', handle_unknown='error')
        return encoder.fit_transform(X[self.feats]).toarray()


    """Apply standardization to numerical features

    Inherits: BaseEstimator, TransformerMixin
    """
class CustomStandardScaler(BaseEstimator, TransformerMixin):

    """Constructor for CustomStandardScaler

    Positional Arguments:
        feats: a list of numerical features to standardize
    """
    def __init__(self, feats):
        self.feats = feats
        

    """fit data to this transformer

    Positional Arguments:
        X - model matrix of predictors
    """
    def fit(self, X, y=None):
        return self
    
    """transform data using centering and scaling

    Positional Arguments:
        X - model matrix of predictors
    """
    def transform(self, X, y=None):
        scaler = StandardScaler()
        return scaler.fit_transform(X[self.feats])

    """pipeline_xgbr: ML pipeline with the following steps

       1. Numerical feature transformations
           1a. Apply standardization
           1b. Apply KNN imputer to missing data
           1c. Calculate 8 principal components

       2. Apply treatment coding to the categorical features
       3. Fit XGBoost to the combined results of (2) and (3)
    """
def pipeline_xgbr(num_feats_to_keep, cat_feats_to_encode):
    pipelinexgb = Pipeline([
           ('features', FeatureUnion([
               ('PCA_pipeline', Pipeline([
                    ('StandardScaler', CustomStandardScaler(num_feats_to_keep)),
                    ('Imputer', KNNImputer()),
                    ('PCA', PCA(n_components=8))
                ])),
                ('factor_encoder', CustomEncoder(cat_feats_to_encode))
            ])),
            ('xgbr', XGBRegressor(random_state=123))
        ])
    return pipelinexgb

    """pipeline_knn: ML pipeline with the following steps

       1. Numerical feature transformations
           1a. Apply standardization
           1b. Apply KNN imputer to missing data
           1c. Calculate 8 principal components

       2. Apply treatment coding to the categorical features
       3. Fit XGBoost to the combined results of (2) and (3)
    """
def pipeline_knn(num_feats_to_keep, cat_feats_to_encode):
    pipelineknn = Pipeline([
        ('features', FeatureUnion([
           ('PCA_pipeline', Pipeline([
                ('StandardScaler', CustomStandardScaler(num_feats_to_keep)),
                ('Imputer', KNNImputer()),
                ('PCA', PCA(n_components=8))
            ])),
            ('factor_encoder', CustomEncoder(cat_feats_to_encode))
        ])),
        ('knn', KNeighborsRegressor())
    ])
    return pipelineknn

    """pipeline_ridge: ML pipeline with the following steps

       1. Numerical feature transformations
           1a. Apply standardization
           1b. Apply KNN imputer to missing data
           1c. Calculate 8 principal components

       2. Apply treatment coding to the categorical features
       3. Fit XGBoost to the combined results of (2) and (3)
    """
def pipeline_ridge(num_feats_to_keep, cat_feats_to_encode):
    pipelineridge = Pipeline([
        ('features', FeatureUnion([
           ('PCA_pipeline', Pipeline([
                ('StandardScaler', CustomStandardScaler(num_feats_to_keep)),
                ('Imputer', KNNImputer()),
                ('PCA', PCA(n_components=8))
            ])),
            ('factor_encoder', CustomEncoder(cat_feats_to_encode))
        ])),
        ('ridge', Ridge())
    ])
    return pipelineridge

    """pipeline_lm: ML pipeline with the following steps

       1. Numerical feature transformations
           1a. Apply standardization
           1b. Apply KNN imputer to missing data
           1c. Calculate 8 principal components

       2. Apply treatment coding to the categorical features
       3. Fit XGBoost to the combined results of (2) and (3)
    """
def pipeline_lm(num_feats_to_keep, cat_feats_to_encode):
    pipelinelm = Pipeline([
        ('features', FeatureUnion([
           ('PCA_pipeline', Pipeline([
                ('StandardScaler', CustomStandardScaler(num_feats_to_keep)),
                ('Imputer', KNNImputer()),
                ('PCA', PCA(n_components=8))
            ])),
            ('factor_encoder', CustomEncoder(cat_feats_to_encode))
        ])),
        ('glm', LinearRegression())
    ])
    return pipelinelm
