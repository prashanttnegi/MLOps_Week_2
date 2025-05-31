import pandas as pd

df = pd.read_csv('data/iris.csv')

df = df[0:50]

df.to_csv('data/iris.csv', index=False)