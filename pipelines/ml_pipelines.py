from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, feats):
        self.feats = feats
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        encoder = OneHotEncoder(drop='first', handle_unknown='error')
        return encoder.fit_transform(X[self.feats]).toarray()

class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feats):
        self.feats = feats
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        scaler = StandardScaler()
        return scaler.fit_transform(X[self.feats])

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
