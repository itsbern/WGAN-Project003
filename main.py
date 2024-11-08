import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from redes.dense_net import Generator, Discriminator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define hiperparámetros
latent_dim = 350
ts_dim = 252
epochs = 1000
n_critic = 5
learning_rate = 0.0001
lambda_gp = 10  # Coeficiente de penalización de gradiente
batch_size = 1

# Crear el generador y discriminador
generator = Generator(latent_dim, ts_dim)
discriminator = Discriminator(ts_dim)

# Optimizadores
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Preprocesamiento de los datos (normalización)
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return data_normalized, scaler

# Postprocesamiento de los datos generados (inversa de la normalización)
def postprocess_data(data, scaler):
    data = data.reshape(-1, 1)  # Asegura que data sea 2D
    data_original_scale = scaler.inverse_transform(data)
    return data_original_scale.flatten()  # Aplana de nuevo para devolver un array 1D

# Función para convertir rendimientos a precios
def convert_returns_to_prices(returns, initial_price):
    """
    Convierte una serie de rendimientos logarítmicos en precios.
    
    Args:
        returns (np.array): Serie de rendimientos logarítmicos.
        initial_price (float): Precio inicial para la reconstrucción de la serie de precios.
        
    Returns:
        np.array: Serie de precios generados.
    """
    prices = initial_price * np.exp(np.cumsum(returns))
    return prices

# Función de penalización de gradiente para WGAN-GP
def gradient_penalty(real_data, fake_data):
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
    alpha = tf.broadcast_to(alpha, real_data.shape)  # Ajustar la forma de alpha
    real_data = tf.cast(real_data, tf.float32)
    fake_data = tf.cast(fake_data, tf.float32)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
    return tf.reduce_mean((grad_norm - 1.0) ** 2)

# Función para entrenar el discriminador con penalización de gradiente
@tf.function
def train_discriminator_with_gp(real_data, lambda_gp, noise):
    current_batch_size = tf.shape(real_data)[0]  # Tamaño del lote actual
    fake_data = generator(noise)

    with tf.GradientTape() as tape:
        real_output = discriminator(real_data)
        fake_output = discriminator(fake_data)
        
        # Pérdida Wasserstein con penalización de gradiente
        d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        gp = gradient_penalty(real_data, fake_data)
        d_loss += lambda_gp * gp
    
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    
    return d_loss

# Función para entrenar el generador
@tf.function
def train_generator(batch_size, noise):
    with tf.GradientTape() as tape:
        fake_data = generator(noise)
        fake_output = discriminator(fake_data)
        g_loss = -tf.reduce_mean(fake_output)

    gradients = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    return g_loss

# Almacenar las métricas de pérdida
d_losses = []
g_losses = []

# Función de entrenamiento
def train(dataset, epochs, batch_size, n_critic):
    for epoch in range(epochs):
        for real_data in dataset:
            real_data = tf.convert_to_tensor(real_data)
            noise = tf.random.normal([batch_size, latent_dim])
            for _ in range(n_critic):
                d_loss = train_discriminator_with_gp(real_data, lambda_gp, noise)
            g_loss = train_generator(batch_size, noise)

        # Almacenar pérdidas cada 100 épocas
        if epoch % 100 == 0:
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
            print(f"Epoch {epoch}, D Loss: {d_loss.numpy()}, G Loss: {g_loss.numpy()}")

# Cargar el archivo CSV y preprocesar los datos
real_price = pd.read_csv('/Users/Usuario/OneDrive/Escritorio/ITESO/MICRO-TRADING/Project-GAN/data/aapl_data.csv')
real_price_values = (np.log(real_price['Close']) - np.log(real_price["Close"].shift())).dropna().values
real_price_normalized, scaler = preprocess_data(real_price_values)

# Crear dataset sintético con los datos preprocesados
data = np.array(real_price_normalized)
num_elements = (len(data) // ts_dim) * ts_dim
data = data[:num_elements]
data = data.reshape(-1, ts_dim)
dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size, drop_remainder=True)

# Entrena la red
train(dataset, epochs, batch_size, n_critic)

# Generar precios sintéticos ajustando la dimensión de salida
def generate_synthetic_prices(generator, num_scenarios, latent_dim, ts_dim, scaler, initial_price):
    scenarios = []
    for _ in range(num_scenarios):
        noise = tf.random.normal([1, latent_dim])
        generated_returns = generator(noise)
        
        # Asegúrate de que la salida tenga el tamaño ts_dim
        generated_returns = tf.reshape(generated_returns, (ts_dim,))
        generated_returns = postprocess_data(generated_returns.numpy(), scaler)
        generated_prices = convert_returns_to_prices(generated_returns, initial_price)
        scenarios.append(generated_prices)
    return np.array(scenarios)

# Generar 100 escenarios y postprocesar para devolverlos a la escala original
num_scenarios = 100
initial_price = real_price['Close'].iloc[0]
synthetic_prices = generate_synthetic_prices(generator, num_scenarios, latent_dim, ts_dim, scaler, initial_price)

# Graficar pérdidas de entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(range(0, epochs, 100), d_losses, label='Discriminator Loss')
plt.plot(range(0, epochs, 100), g_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()

for i in range(num_scenarios):
    plt.plot(synthetic_prices[i], alpha=0.3)
plt.show()

# Graficar precio real vs precios generados en la escala original
plt.figure(figsize=(10, 5))
plt.plot(real_price['Close'].values[:252], label='Real Price', color='black', linewidth=2)
for i in range(num_scenarios):
    plt.plot(synthetic_prices[i], alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Real Price vs Synthetic Prices')
plt.legend(['Real Price'])
plt.show()
