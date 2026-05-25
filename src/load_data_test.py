import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/fred_md.csv")

print(df.shape)

print(df.columns[:10])

print(df.iloc[0])

print(df.iloc[1])

print("success 1")

data = df.iloc[1:]
data['sasdate'] = pd.to_datetime(data['sasdate'])
data = data.set_index('sasdate')

data['RPI'].plot(title='Real Personal Income', ylabel='$', xlabel='Date')
plt.show()

print('success')
