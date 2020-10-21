import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
x=pd.read_csv("9.1 car-sales-extended-missing-data.csv.csv")
x
x.head()
x.shape
x.describe()
x.describe(include='all')
x.isnull()
x.isnull().sum()
x["Price"].plot.bar()