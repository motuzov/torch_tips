import urllib

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger


class Lstm(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=50,
            num_layers=1,
            batch_first=True
            )
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    #def train_dataloader(self) -> TRAIN_DATALOADERS:
    #    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    #    return loader

    def training_step(self, train_batch, batch_idx):
        features, y = train_batch
        pred = self(features)
        loss = F.mse_loss(pred, y)
        #self.log('train_loss',
        #         torch.sqrt(loss)
        #         )
        return loss


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


def load_data():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    f = urllib.request.urlopen(url)
    df = pd.read_csv(f)
    return df


def train(timeseries, split_ratio=0.67):
    lookback = 1
    n = len(timeseries)
    n_train = int(n*split_ratio)
    train, test = (
        timeseries[:n_train],
        timeseries[n_train:]
    )
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    train_loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=8
    )
    val_loader = data.DataLoader(
        data.TensorDataset(X_test, y_test),
        shuffle=True,
        batch_size=8
    )

    model = Lstm()
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=2000,
        logger=logger
        )
    trainer.fit(model, train_loader, val_loader)
    with torch.no_grad():
        test_y_pred = model(X_test)
        test_y_pred = test_y_pred.flatten().cpu().numpy()
        test_plot = np.empty(n)
        test_plot[:] = np.nan
        test_plot[n_train+lookback:] = test_y_pred

        train_y_pred = model(X_train)
        train_y_pred = train_y_pred.flatten().cpu().numpy()
        train_plot = np.empty(n)
        train_plot[:] = np.nan
        train_plot[:n_train-lookback] = train_y_pred

        plt.plot(timeseries)
        plt.plot(train_plot)
        plt.plot(test_plot)


if __name__ == '__main__':
    df = load_data()
    timeseries = df[["Passengers"]].values.astype('float32')
    plt.plot(timeseries)
    # plt.show()
    train(timeseries)
    print('!')
