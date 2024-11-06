import tensorflow as tf
from redes.dense_net_skip import Generator, Discriminator  # Se asume que estas clases están adaptadas a TensorFlow
from entrenamiento import Trainer  # Clase Trainer en versión TensorFlow (de la respuesta anterior)
import os
import datetime

latent_dim = 10
ts_dim = 23
conditional = 3

# Define rutas de salida para los resultados
time = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
formatted_time = time.strftime("%Y-%m-%dT%H-%M")  # Formato de tiempo sin caracteres no permitidos

scorepath = f"/Users/Usuario/OneDrive/Escritorio/ITESO/MICRO-TRADING/Project-GAN/results/result_{formatted_time}"
plot_scorepath = os.path.join(scorepath, "line_generation")
os.makedirs(scorepath, exist_ok=True)
os.makedirs(plot_scorepath, exist_ok=True)

# Instancia los modelos
generator = Generator(latent_dim=latent_dim, ts_dim=ts_dim, condition=conditional)
discriminator = Discriminator(ts_dim=ts_dim)

# Rutas para cargar modelos preentrenados (descomentar si necesitas cargar los modelos)
# generator.load_weights("/path/to/gen_CWGAN-GP_model_Dense3_concat_fx.h5")
# discriminator.load_weights("/path/to/dis_CWGAN-GP_model_Dense3_concat_fx.h5")

# Inicialización de los optimizadores
lr_a = 1e-4
lr_b = 1e-4

# Adam y RMSprop son compatibles con TensorFlow, así que los podemos usar directamente
G_opt = tf.keras.optimizers.RMSprop(learning_rate=lr_a)
D_opt = tf.keras.optimizers.RMSprop(learning_rate=lr_b)

# Programadores de tasa de aprendizaje en TensorFlow
def cyclic_lr(base_lr, max_lr, step_size):
    def scheduler(epoch):
        cycle = tf.math.floor(1 + epoch / (2 * step_size))
        x = tf.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * tf.maximum(0., (1 - x))
        return lr
    return scheduler

# Crear programadores de aprendizaje para generador y discriminador
D_scheduler = tf.keras.callbacks.LearningRateScheduler(cyclic_lr(base_lr=1e-4, max_lr=8e-4, step_size=100))
G_scheduler = tf.keras.callbacks.LearningRateScheduler(cyclic_lr(base_lr=1e-4, max_lr=6e-4, step_size=100))

epochs = 1000
batch_size = 128
use_cuda = tf.config.list_physical_devices('GPU')
print("CUDA disponible:", use_cuda)

# Instancia la clase Trainer con los parámetros convertidos a TensorFlow
train = Trainer(generator, discriminator, G_opt, D_opt, batch_size, scorepath, ts_dim, latent_dim, D_scheduler, G_scheduler, use_cuda=bool(use_cuda))
train.train(epochs=epochs)
