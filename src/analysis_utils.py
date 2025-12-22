import  pandas as pd
import numpy as np

def compute_yearly_averages(df):
    yearly_averages = []
    for year in df.index:
        average = df.loc[year].mean()
        yearly_averages.append(average)
    return yearly_averages

