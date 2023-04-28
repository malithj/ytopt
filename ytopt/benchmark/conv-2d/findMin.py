import numpy as np
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

dataframe = pandas.read_csv("results.csv")
print(dataframe)
array = dataframe.loc[:, 'objective']
x = array.to_numpy()

print("Performance summary based on", len(array), "evaluations:")
print("Min: ", x.min(), "s")
print("Max: ", x.max(), "s")
print("Mean: ", x.mean(), "s")
print("The best configurations (for the smallest time) of P0, P1, P2, P3, P4, P5, P6, P7, P8, and P9 is:\n")
print("P0  P1  P2  P3  P4  P5  P6  P7  P8  P9	execution time	     elapsed time\n")
print(dataframe.iloc[np.argmin(x),:])
