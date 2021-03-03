#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
from IPython.display import display
class Knn:
    def __init__(self):
        pass
    def predict(self, data, new_point, k, class_label, p=2 ):
        """data is a pandas.dataframe with class_label in first column and 
        data rows for all the rest as numerical for ex. eucliean distance"""
        data = data.copy()
        distances = [distance(x_row.drop(class_label).numpy(), new_point, p) for index, x_row in data.iterrows()] ##. todo x_row
        data['distances']=distances
        data.distances.sort(ascending=True)
        k_first = data.iloc[0:k]
        classification = k_first.class_label.mode() ## pseudocode
        return classification
    @staticmethod
    def distance(x1, x2, p=2):
        terms = [abs(x1i-x2i)**p for x1i, x2i in zip(x1, x2)]
        summed = sum(term for term in terms) ## todo syntax
        distance_metric = summed**(1/p)
        return distance_metric

