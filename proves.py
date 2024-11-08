import tensorflow as tf

from redes.dense_net import Generator, Discriminator


# Define un lote de datos simulados con la misma forma que tus datos reales
batch_size = 64
ts_dim = 10
latent_dim = 100


generator = Generator(latent_dim, ts_dim)
discriminator = Discriminator(ts_dim)

def gradient_penalty(real_data, fake_data):
    alpha = tf.random.uniform([batch_size, ts_dim], 0.0, 1.0, dtype=tf.float32)
    real_data = tf.cast(real_data, tf.float32)
    fake_data = tf.cast(fake_data, tf.float32)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
    return tf.reduce_mean((grad_norm - 1.0) ** 2)

# Crea datos reales y generados simulados con la forma esperada
real_data = tf.random.normal([batch_size, ts_dim], dtype=tf.float32)
fake_data = tf.random.normal([batch_size, ts_dim], dtype=tf.float32)

# Llama a la función gradient_penalty y guarda el resultado
gp_result = gradient_penalty(real_data, fake_data)

# Imprime el resultado para verificar
print("Penalización de gradiente:", gp_result.numpy())