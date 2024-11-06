import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(self, length, path):
        self.data = pd.read_csv(path, skiprows=[1])
        self.delta = 0.6
        self.length = length
        self.init_all()
        
    def init_all(self):
        self.get_scalar()
        self.store_price()
        self.bid_return = self.preprocessing(self.data['Close'])
        self.store_dates()
        self.data_augment()
    
    def store_price(self):
        self.bid = self.data['Close'].to_numpy()
        
    def store_dates(self):
        self.dates = pd.to_datetime(self.data['date'])
        
    def get_scalar(self):
        self.scalar = StandardScaler()
        self.scalar2 = StandardScaler()
        
    def moving_window(self, x, length):
        return [x[i: i + length] for i in range(0, (len(x) + 1) - length, 4)]
    
    def preprocessing(self, data):
        log_returns = np.log(data / data.shift(1)).fillna(0).to_numpy().reshape(-1, 1)
        self.scalar.fit(log_returns)
        log_returns = self.scalar.transform(log_returns).squeeze()
    
        # Cambiar a log_returns_smoothed
        log_returns_smoothed = np.log(1 + np.abs(log_returns)) * np.sign(log_returns)
        log_returns_smoothed = log_returns_smoothed.reshape(-1, 1)
        self.scalar2.fit(log_returns_smoothed)
        log_returns_smoothed = self.scalar2.transform(log_returns_smoothed).squeeze()
    
        return log_returns_smoothed

    
    def data_augment(self):
        self.bid_return_aug = np.array(self.moving_window(self.bid_return, self.length))
        self.bid_aug = np.array(self.moving_window(self.bid, self.length))
        self.dates_aug = np.array(self.moving_window(self.dates, self.length))
        
    def post_processing(self, return_data, init):
        return_data = self.scalar2.inverse_transform(return_data.reshape(-1, 1)).flatten()
        return_data = return_data * np.exp(0.5 * self.delta * return_data**2)
        return_data = self.scalar.inverse_transform(return_data.reshape(-1, 1)).flatten()
        return_data = np.exp(return_data)
        
        post_return = np.empty((return_data.shape[0],))
        post_return[0] = init
        for i in range(1, return_data.shape[0]):
            post_return[i] = post_return[i - 1] * return_data[i]
            
        return post_return

    def get_samples(self, G, latent_dim, n, ts_dim, conditional, use_cuda=False):
    # Genera ruido aleatorio para los datos
        noise = tf.random.normal((n, 1, latent_dim))
        idx = np.random.randint(self.bid_return_aug.shape[0], size=n)

    # Obtiene muestras reales de los datos
        real_samples = self.bid_return_aug[idx, :]
        real_start_prices = self.bid_aug[idx, 0]
        real_samples = np.expand_dims(real_samples, axis=1)
        real_samples = tf.convert_to_tensor(real_samples, dtype=tf.float32)

    # Si el modelo es condicional, concatena los datos reales al ruido en lugar de asignar directamente
        if conditional > 0:
            noise_conditional_part = real_samples[:, :, :conditional]
            noise_random_part = noise[:, :, conditional:]
            noise = tf.concat([noise_conditional_part, noise_random_part], axis=-1)

    # Genera los datos con el modelo
        y = G(noise)

    # Concatenamos los datos generados y los reales si es condicional
        y = tf.concat([real_samples[:, :, :conditional], y], axis=2)

        return y, real_samples, real_start_prices
