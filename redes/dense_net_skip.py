import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, latent_dim, ts_dim, condition):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.ts_dim = ts_dim
        self.condition = condition
        self.hidden = 128
        
        # Define the blocks in TensorFlow
        self.block = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_shape=(256,)),
            tf.keras.layers.LeakyReLU()
        ])
        
        self.block_cnn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(self.hidden, kernel_size=3, dilation_rate=2, padding='same'),
            tf.keras.layers.LeakyReLU()
        ])
        
        self.block_shift = tf.keras.Sequential([
            tf.keras.layers.Conv1D(10, kernel_size=3, dilation_rate=2, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU()
        ])
        
        self.noise_to_latent = tf.keras.Sequential([
            tf.keras.layers.Conv1D(self.hidden, kernel_size=1, input_shape=(None, self.latent_dim)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(self.hidden, kernel_size=5, dilation_rate=2, padding='same'),
            tf.keras.layers.LeakyReLU()
        ])
        
        self.latent_to_output = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ts_dim - self.condition)
        ])

    def call(self, input_data):
        x = self.noise_to_latent(input_data)
        x_block = self.block_cnn(x)
        x = x_block + x  # Residual connection
        x_block = self.block_cnn(x)
        x = x_block + x
        x_block = self.block_cnn(x)
        x = x_block + x
        x = self.block_shift(x)
        x_block = self.block(x)
        x = x_block + x  # Residual connection
        x_block = self.block(x)
        x = x_block + x
        x_block = self.block(x)
        x = x_block + x
        x = self.latent_to_output(x)
        return tf.expand_dims(x, axis=1)


class Discriminator(tf.keras.Model):
    def __init__(self, ts_dim):
        super(Discriminator, self).__init__()

        self.ts_dim = ts_dim
        
        # Define layers
        self.ts_to_feature = tf.keras.Sequential([
            tf.keras.layers.Dense(512, input_shape=(self.ts_dim,)),
            tf.keras.layers.LeakyReLU()
        ])
        
        self.block = tf.keras.Sequential([
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU()
        ])
        
        self.to_score = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
            # layers.Activation('sigmoid') # Uncomment if needed for binary output
        ])

    def call(self, input_data):
        x = self.ts_to_feature(input_data)
        x_block = self.block(x)
        x = x + x_block  # Residual connection
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x_block = self.block(x)
        x = x + x_block
        x = self.to_score(x)
        return x
