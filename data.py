import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.special import lambertw
import matplotlib.pyplot as plt






class DataProcessor:
    def __init__(self, length, path, delta=0.6):
        self.length = length
        self.path = path
        self.delta = delta
        self.scalar = StandardScaler()
        self.scalar2 = StandardScaler()
        self.data = pd.read_csv(path)
        self.initialize_data()

    def initialize_data(self):
        self.store_price()
        self.bid_return = self.preprocess_returns(self.data['Close'])
        self.augment_data()

    def store_price(self):
        self.bid = self.data['Close'].to_numpy()

    def moving_window(self, x, length, step=4):
        return np.array([x[i:i + length] for i in range(0, len(x) - length + 1, step)])

    def preprocess_returns(self, data):
        log_returns = np.log(data / data.shift(1)).fillna(0).to_numpy().reshape(-1, 1)
        log_returns = self.scalar.fit_transform(log_returns).flatten()
        
        # Lambert W transformation
        log_returns_w = np.sign(log_returns) * np.sqrt(lambertw(self.delta * log_returns ** 2) / self.delta).real
        log_returns_w = self.scalar2.fit_transform(log_returns_w.reshape(-1, 1)).flatten()
        return log_returns_w

    def augment_data(self):
        self.bid_return_aug = self.moving_window(self.bid_return, self.length)
        self.bid_aug = self.moving_window(self.bid, self.length)

    def post_process(self, return_data, init_price):
        return_data = self.scalar2.inverse_transform(return_data.reshape(-1, 1)).flatten()
        return_data = return_data * np.exp(0.5 * self.delta * return_data ** 2)
        return_data = self.scalar.inverse_transform(return_data.reshape(-1, 1)).flatten()
        return_data = np.exp(return_data)
        
        post_price = np.empty_like(return_data)
        post_price[0] = init_price
        for i in range(1, len(return_data)):
            post_price[i] = post_price[i - 1] * return_data[i]
        return post_price

    def get_batch(self, batch_size):
        idx = np.random.randint(0, len(self.bid_return_aug), batch_size)
        return self.bid_return_aug[idx], self.bid_aug[idx, 0]
