import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

class LoadData():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.tensor(self.X.iloc[index])
        y = torch.tensor(self.y.iloc[index])
        return X, y

def dataloader_generate(data_path, anomaly_known, cont_rate, batch_size):
    # Load data
    df = pd.read_csv(data_path, index_col=None)
    dim = len(df.columns)
    x = df[df.columns[0:dim - 1]].copy()
    y = df[df.columns[dim - 1]].copy()

    # Split training set: validation set: test set = 6:2:2
    x_train, x_remain, y_train, y_remain = train_test_split(x, y, test_size=0.4, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_remain, y_remain, test_size=0.5, random_state=42)

    train_set = pd.concat([x_train, y_train], axis=1)
    print(train_set.shape)
    dim = len(train_set.columns)

    # Uniformly change the last column to 'label'
    train_set = train_set.rename(columns={df.columns[-1]: 'label'})
    N = train_set[train_set['label'] == 0]
    A = train_set[train_set['label'] == 1]

    # If the number of anomalies is too small, directly sample all anomalies
    if len(A) < anomaly_known:
        anomaly_known = len(A)
    A_set = A.sample(n=anomaly_known, replace=False, random_state=42)
    A = A.drop(A_set.index)

    cont_num = round(len(y_train[y_train == 0]) / (1 - cont_rate) * cont_rate)
    if cont_num < len(A):
        cont_set = A.sample(n=cont_num, replace=False, random_state=42)
    else:
        cont_set = A

    U_set = pd.concat([N, cont_set], axis=0, ignore_index=True)
    print(U_set['label'].value_counts())

    A_set['label'] = 1
    U_set['label'] = 0

    train_set = U_set
    x_train = train_set[train_set.columns[0:dim - 1]]
    y_train = train_set[train_set.columns[dim - 1]]

    X_dimension = len(x_train.columns)
    y_dimension = len(y_train.value_counts())
    print(f"X dimension：{X_dimension}")
    print(f"y dimension：{y_dimension}")

    train_data = LoadData(x_train, y_train)
    test_data = LoadData(x_test, y_test)
    val_data = LoadData(x_val, y_val)

    trainset_len = len(train_data)

    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size)
    val_loader = Data.DataLoader(val_data, batch_size=batch_size)

    # compute the mean vector of A and U
    A_set_np = A_set.values[:, :-1]
    A_set_np = A_set_np.astype(float)
    mean_a = torch.mean(torch.tensor(A_set_np), dim=0)

    U_set_np = U_set.values[:, :-1]
    U_set_np = U_set_np.astype(float)
    mean_u = torch.mean(torch.tensor(U_set_np), dim=0)

    return train_loader, test_loader, val_loader, y_test, y_val, A_set, U_set, mean_a, mean_u, dim, trainset_len
