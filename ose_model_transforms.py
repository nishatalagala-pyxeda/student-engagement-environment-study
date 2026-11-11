"""Serializable training-only facial dimension reduction for OSE models."""
import numpy as np
import joblib
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

@joblib.Memory('/private/tmp/ose-face-preprocessing-cache',verbose=0).cache
def fit_facial_block(X,n_components):
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True);a=imputer.fit_transform(X);scaler=StandardScaler().fit(a);a=scaler.transform(a);reducer=None
    if np.any(np.var(a,axis=0)>0):reducer=PCA(n_components=min(n_components,len(a)-1,a.shape[1]),svd_solver='randomized',random_state=42).fit(a)
    return imputer,scaler,reducer

class FacialReducer(BaseEstimator,TransformerMixin):
    def __init__(self,n_components=10):self.n_components=n_components
    def fit(self,X,y=None):
        self.imputer_,self.scaler_,self.reducer_=fit_facial_block(X,self.n_components);return self
    def transform(self,X):
        a=self.scaler_.transform(self.imputer_.transform(X));out=np.zeros((len(a),self.n_components))
        if self.reducer_ is not None:
            z=self.reducer_.transform(a);out[:,:z.shape[1]]=z
        return out
