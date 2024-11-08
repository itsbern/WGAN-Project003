import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.special import lambertw
import matplotlib.pyplot as plt




# Entrenamiento WGAN-GP
class WGAN_GP_Trainer:
    def __init__(self, generator, discriminator, data_processor, latent_dim, batch_size, gp_weight, n_critic, learning_rate):
        self.generator = generator
        self.discriminator = discriminator
        self.data_processor = data_processor
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.gp_weight = gp_weight
        self.n_critic = n_critic
        self.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.d_losses = []
        self.g_losses = []

    def gradient_penalty(self, real_data, fake_data):
        alpha = tf.random.uniform([self.batch_size, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated)
        grads = tape.gradient(pred, interpolated)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        return tf.reduce_mean((grad_norm - 1.0) ** 2)

    def train_discriminator(self, real_data):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        fake_data = self.generator(noise)
        
        with tf.GradientTape() as tape:
            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data)
            gp = self.gradient_penalty(real_data, fake_data)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + self.gp_weight * gp
        
        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return d_loss

    def train_generator(self):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data)
            g_loss = -tf.reduce_mean(fake_output)
        
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return g_loss

    def train(self, epochs):
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                real_batch, _ = self.data_processor.get_batch(self.batch_size)
                real_batch = tf.convert_to_tensor(real_batch, dtype=tf.float32)
                d_loss = self.train_discriminator(real_batch)
            g_loss = self.train_generator()

            if epoch % 100 == 0:
                self.d_losses.append(d_loss.numpy())
                self.g_losses.append(g_loss.numpy())
                print(f"Epoch {epoch}, D Loss: {d_loss.numpy()}, G Loss: {g_loss.numpy()}")

        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, epochs, 100), self.d_losses, label='Discriminator Loss')
        plt.plot(range(0, epochs, 100), self.g_losses, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.show()

