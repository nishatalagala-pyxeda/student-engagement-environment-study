"""Portable native CatBoost input conversion and training-only facial reduction."""
import pandas as pd
from ose_model_transforms import FacialReducer

class NativeInput:
    def __init__(self,columns,schema):
        self.columns=columns
        self.face=[c for c in columns if c in schema['face_features']]
        self.cats=[c for c in columns if c in schema['categorical_features']]
        self.nums=[c for c in columns if c not in self.face+self.cats]
        self.cat_indices=list(range(len(self.nums),len(self.nums)+len(self.cats)))
    def fit(self,X):
        self.facial=FacialReducer().fit(X[self.face]);return self
    def transform(self,X):
        parts=[X[self.nums].reset_index(drop=True).astype(float),X[self.cats].reset_index(drop=True).fillna('Unknown').astype(str),pd.DataFrame(self.facial.transform(X[self.face]))]
        out=pd.concat(parts,axis=1);out.columns=['f'+str(i) for i in range(out.shape[1])];return out
