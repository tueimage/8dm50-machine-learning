# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:39:17 2022

@author: s139188
"""
import pandas as pd
filename = 'QSAR_2.csv'   # nelem=41.6k  (n=1300, p=32)
df = pd.read_csv(filename)

# df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
#      index=['cobra', 'viper', 'sidewinder'],
#      columns=['max_speed', 'shield'])
print(df)
df = df.loc[:, :]
print(df)