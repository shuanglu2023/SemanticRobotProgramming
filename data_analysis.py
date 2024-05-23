import os
import pandas as pd
import numpy as np
# read csv file
file_name = "estimated_depth"
# list all files with file_name in the directory
files = [f for f in os.listdir() if file_name in f]

for file in files:
    # read csv file in dataframe
    df = pd.read_csv(file)
    # calcualte the mean squared error between the first and last column, if the entry is not zero in the first column
    mse = 0
    count = 0
    for index, row in df.iterrows():
        if row[0] != 0:
            mse += (row[0]-row[len(row)-1])**2
            count += 1
    # print(np.sqrt(mse/count))
    # percentage of non-zero entries in the first column
    print(file, count/len(df))