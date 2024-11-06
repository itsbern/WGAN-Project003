from data import Data
import tensorflow as tf
import numpy as np
from eval_u import plt_loss, plt_progress, plt_gp, plt_lr  # Asegúrate de tener estas funciones adaptadas a TensorFlow

class Trainer:
    def __init__(self, generator, critic, gen_optimizer, critic_optimizer, batch_size, path, ts_dim, latent_dim, D_scheduler, G_scheduler, gp_weight=10, critic_iter=5, n_eval=20, use_cuda=False):
        self.G = generator
        self.D = critic
        self.G_opt = gen_optimizer
        self.D_opt = critic_optimizer
        self.G_scheduler = G_scheduler
        self.D_scheduler = D_scheduler
        self.batch_size = batch_size
        self.scorepath = path
        self.gp_weight = gp_weight
        self.critic_iter = critic_iter
        self.n_eval = n_eval
        self.use_cuda = use_cuda
        self.conditional = 3
        self.ts_dim = ts_dim
        data_load_path = '/Users/Usuario/OneDrive/Escritorio/ITESO/MICRO-TRADING/Project-GAN/data/aapl_data.csv'  
        self.data = Data(self.ts_dim, data_load_path)
        self.latent_dim = latent_dim
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'LR_G': [], 'LR_D': []}

    def train(self, epochs):
        plot_num = 0
        for epoch in range(epochs):
            for i in range(self.critic_iter):
                # Obtener muestras reales y falsas
                fake_batch, real_batch, start_prices = self.data.get_samples(
                    G=self.G, latent_dim=self.latent_dim, n=self.batch_size, ts_dim=self.ts_dim, conditional=self.conditional, use_cuda=self.use_cuda
                )

                # Ajustar la forma de los datos si es necesario
                real_batch = tf.squeeze(real_batch, axis=1)
                fake_batch = tf.squeeze(fake_batch, axis=1)

                # Entrenamiento del discriminador
                with tf.GradientTape() as tape:
                    d_real = self.D(real_batch, training=True)
                    d_fake = self.D(fake_batch, training=True)
                    grad_penalty, grad_norm = self._grad_penalty(real_batch, fake_batch)
                    d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + grad_penalty

                # Aplicar gradientes
                grads = tape.gradient(d_loss, self.D.trainable_variables)
                self.D_opt.apply_gradients(zip(grads, self.D.trainable_variables))
                
                # Actualizar registros de pérdida
                if i == self.critic_iter - 1:
                    self.losses['LR_D'].append(self.D_opt.learning_rate.numpy())
                    self.losses['D'].append(float(d_loss))
                    self.losses['GP'].append(float(grad_penalty))
                    self.losses['gradient_norm'].append(float(grad_norm))

            # Entrenamiento del generador
            with tf.GradientTape() as tape:
                fake_batch_critic, _, _ = self.data.get_samples(
                    G=self.G, latent_dim=self.latent_dim, n=self.batch_size, ts_dim=self.ts_dim, conditional=self.conditional, use_cuda=self.use_cuda
                )
                fake_batch_critic = tf.squeeze(fake_batch_critic, axis=1)
                d_critic_fake = self.D(fake_batch_critic, training=True)
                g_loss = -tf.reduce_mean(d_critic_fake)
            
            grads = tape.gradient(g_loss, self.G.trainable_variables)
            self.G_opt.apply_gradients(zip(grads, self.G.trainable_variables))

            # Actualizar registro de tasa de aprendizaje y pérdidas
            self.losses['LR_G'].append(self.G_opt.learning_rate.numpy())
            self.losses['G'].append(float(g_loss))
            print(f"Epoch: {epoch + 1}/{epochs}, G_loss: {g_loss}, D_loss: {d_loss}, GP: {grad_penalty}")

            # Guardar visualizaciones y evaluar
            if (epoch + 1) % self.n_eval == 0:
                if (epoch + 1) % 1000 == 0:
                    plot_num += 1
                plt_loss(self.losses['G'], self.losses['D'], self.scorepath, plot_num)
                plt_gp(self.losses['gradient_norm'], self.losses['GP'], self.scorepath)
                plt_lr(self.losses['LR_G'], self.losses['LR_D'], self.scorepath)
                
            if (epoch + 1) % (10 * self.n_eval) == 0:
                fake_lines, real_lines, start_prices = self.data.get_samples(
                    G=self.G, latent_dim=self.latent_dim, n=4, ts_dim=self.ts_dim, conditional=self.conditional, use_cuda=self.use_cuda
                )
                real_lines = np.squeeze(real_lines.numpy())
                fake_lines = np.squeeze(fake_lines.numpy())
                real_lines = np.array([self.data.post_processing(real_lines[i], start_prices[i]) for i in range(real_lines.shape[0])])
                fake_lines = np.array([self.data.post_processing(fake_lines[i], start_prices[i]) for i in range(real_lines.shape[0])])
                plt_progress(real_lines, fake_lines, epoch, self.scorepath)

            if (epoch + 1) % 500 == 0:
                name = 'CWGAN-GP_model_Dense3_concat_fx'
                self.G.save_weights(f"{self.scorepath}/gen_{name}.weights.h5")
                self.D.save_weights(f"{self.scorepath}/dis_{name}.weights.h5") 


    def _grad_penalty(self, real_data, gen_data):
        batch_size = real_data.shape[0]
        t = tf.random.uniform((batch_size, 1), 0.0, 1.0)
        interpol = t * real_data + (1 - t) * gen_data

        with tf.GradientTape() as tape:
            tape.watch(interpol)
            prob_interpol = self.D(interpol, training=True)

        gradients = tape.gradient(prob_interpol, interpol)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]) + 1e-10)
        gradient_penalty = self.gp_weight * tf.reduce_mean(tf.square(tf.maximum(gradients_norm - 1, 0)))

        return gradient_penalty, tf.reduce_mean(gradients_norm)
