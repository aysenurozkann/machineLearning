# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:04:00 2021

@author: aysenurozkan
"""

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


read_data = pd.read_csv(r'housing.csv')

# is there any nan value ?
check_data = read_data.isnull().values.any()
print(check_data, end='\n\n')

# nan variables return True (True = 1)
check_data = read_data.isnull()

# sum the True values and find the number of nan values in the columns
sum_true_variable = check_data.sum()
print(sum_true_variable, end='\n\n')


X = read_data.iloc[:, 2:8].values  # these are indipendent variables
y = read_data.iloc[:, -2].values  # these are dependent variables


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
impute = imputer.fit(X[:, 2:8])
X[:, 2:8] = imputer.transform(X[:, 2:8])

print(X, end='\n\n')
print(np.isnan(X.any()), end='\n\n')  # is there any True (nan) value ?

