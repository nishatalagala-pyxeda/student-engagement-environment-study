"""Train-only transformations for explicit baseline-stress/environment prediction interactions."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

MEANS=['temperature_C_week_mean','humidity_week_mean','co2_week_mean','voc_week_mean','pm25_week_mean','light_week_mean','noise_week_mean']
TRANSFORMED=['scaled_'+c for c in MEANS]
TERMS=['prior_x_'+c for c in TRANSFORMED]


def transformed_levels(frame):
    z=frame[MEANS].to_numpy(float).copy();z[:,1]/=10;z[:,2]/=100
    for j in [3,4,5]:z[:,j]=np.log2(1+z[:,j])
    z[:,6]/=5
    return z


class ModerationTerms(BaseEstimator,TransformerMixin):
    def __init__(self,environment=False,interactions=False):
        self.environment=environment;self.interactions=interactions
    def fit(self,X,y=None):
        values=pd.to_numeric(X.prior_stress,errors='coerce');self.history_median_=float(values.median()) if values.notna().any() else 2.0
        if self.environment:
            z=transformed_levels(X);self.environment_center_=np.mean(z,axis=0);self.environment_scale_=np.std(z,axis=0);self.environment_scale_=np.where(self.environment_scale_>0,self.environment_scale_,1)
        return self
    def transform(self,X):
        d=X.copy();b=pd.to_numeric(d.prior_stress,errors='coerce').fillna(self.history_median_).to_numpy(float)-2.0
        # The same quadratic baseline term is present in every comparator, guarding against a
        # simple nonlinear history relationship being mistaken for environmental moderation.
        d['prior_stress_squared']=b*b
        if self.environment:
            z=transformed_levels(d)
            for j,name in enumerate(TRANSFORMED):d[name]=z[:,j]
            if self.interactions:
                terms=(z-self.environment_center_)/self.environment_scale_*b[:,None]
                for j,name in enumerate(TERMS):d[name]=terms[:,j]
        return d
