import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# input the name of dataset
data_name = 'satellite'

# '.npz' to '.csv
def npz_to_csv(data_name):
    dataset = np.load(data_name + '.npz')
    df1 = pd.DataFrame(dataset['X'])
    df2 = pd.DataFrame(dataset['y'])

    # Change the column name of the label column to 'label'
    c_name = df2.columns[-1]
    df2.rename(columns={c_name: 'label'}, inplace=True)
    df = pd.concat([df1, df2], axis=1, ignore_index=False)
    df.to_csv(data_name + '.csv', index=False)

# npz_to_csv(data_name)

# Load dataset
df = pd.read_csv(data_name + '.csv', sep=',', index_col=None)
# shuffle
df = df.sample(frac=1, replace=False, ignore_index=True)


# Split feature and label
dim = len(df.columns)
df1 = df[df.columns[0:dim-1]]
df2 = df[df.columns[dim-1]]

# Min-Max Normalization
df1 = (df1 - df1.min()) / (df1.max() - df1.min())
df1 = df1.fillna(value=0.0)

df = pd.concat([df1, df2], axis=1, ignore_index=False)

# Change the column name of the label column to 'label'
c_name = df.columns[-1]
print(c_name)
df.rename(columns={c_name: 'label'}, inplace=True)


print(df.shape)
print(df['label'].value_counts())
df.to_csv(data_name + '.csv', index=False)

