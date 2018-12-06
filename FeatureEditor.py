import pandas as pd
import numpy as np
import re

non_numeric= lambda s:re.search('[^0-9.]+',s)!=None

get_non_numeric_cols = lambda df:[col for col in df.columns if df[col][df[col].apply(non_numeric)].size>0]

def LabelCoding(df,column_name,col_dict):
	for key in col_dict:
		df[column_name].replace(key,col_dict[key],inplace=True)

def LabelCoding_Series(df,columns):
    for col in columns:
        symbols= list(df[col].unique())
        symbols.sort()
        df[col].replace(to_replace=symbols,value=list(np.arange(0,len(symbols),1)),inplace=True)

def column_standardization(df,columns=[]):
    for col in columns:
        df[col]= (df[col]-df[col].mean())/df[col].std()

def ensure_numeric(df,columns=[],default_value=-1):
    for col in columns:
        values_nn = df[col][df[col].apply(non_numeric)].unique()
        df[col].replace(to_replace=values_nn,value=-1,inplace=True)
    