import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K

  
# SHARED SAMPLING LAYER
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

  
# 1. EASY TASK VAE (Simple)
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

def build_simple_vae(input_dim):
    
    latent_dim = 16 
    
    # ENCODER
    encoder_inputs = keras.Input(shape=(input_dim,))
    
    x = layers.Dense(256)(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # DECODER
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(128)(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    return encoder, decoder

  
# 2. MEDIUM TASK VAE (Hybrid Audio+Text)
  
class HybridVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(HybridVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        aud, txt = data[0], data[1]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([aud, txt])
            recon_aud, recon_txt = self.decoder(z)
            loss_aud = tf.reduce_mean(tf.reduce_sum(tf.square(aud - recon_aud), axis=1))
            loss_txt = tf.reduce_mean(tf.reduce_sum(tf.square(txt - recon_txt), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total = loss_aud + loss_txt + kl_loss
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total)
        return {"loss": self.total_loss_tracker.result()}

def build_hybrid_vae():
    latent_dim = 32
    input_audio = keras.Input(shape=(26, 1))
    x1 = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(input_audio)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(x1)
    x1 = layers.Flatten()(x1)
    input_text = keras.Input(shape=(768,))
    x2 = layers.Dense(256, activation="relu")(input_text)
    merged = layers.Concatenate()([x1, x2])
    h = layers.Dense(128, activation="relu")(merged)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model([input_audio, input_text], [z_mean, z_log_var, z])
    latent_inputs = keras.Input(shape=(latent_dim,))
    h_dec = layers.Dense(128, activation="relu")(latent_inputs)
    d1 = layers.Dense(26 * 1, activation="sigmoid")(h_dec)
    d1 = layers.Reshape((26, 1))(d1)
    d2 = layers.Dense(768, activation="linear")(h_dec)
    decoder = keras.Model(latent_inputs, [d1, d2])
    return encoder, decoder

  
# 3. HARD TASK CVAE (Conditional)
  
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        # KL Weight starts at 0.0
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        # Unpack data
        if isinstance(data, tuple): data = data[0]
        aud, txt, cond = data[0], data[1], data[2]
        
        with tf.GradientTape() as tape:
            # Forward Pass
            zm, zlv, z = self.encoder([aud, txt, cond])
            ra, rt = self.decoder([z, cond])
            
            # Reconstruction Loss (Using Mean to balance dimensions)
            loss_aud = tf.reduce_mean(tf.square(aud - ra)) 
            loss_txt = tf.reduce_mean(tf.square(txt - rt))
            recon_loss = loss_aud + loss_txt
            
            # KL Divergence
            kl_loss = -0.5 * tf.reduce_mean(1 + zlv - tf.square(zm) - tf.exp(zlv))
            
            # Total Loss (Annealing applied here)
            total_loss = recon_loss + (self.kl_weight * kl_loss)
            
        # Backprop
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        
        new_weight = tf.minimum(self.kl_weight + 0.0001, 0.1)
        self.kl_weight.assign(new_weight)

            
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

def build_cvae(cond_dim=2): 
    latent_dim = 32
    # Encoder Inputs
    in_aud = keras.Input(shape=(26, 1))
    in_txt = keras.Input(shape=(768,))
    in_cond = keras.Input(shape=(cond_dim,)) 
    
    # Encoder Layers
    x1 = layers.Flatten()(layers.Conv1D(32, 3, activation="relu", padding="same")(in_aud))
    x2 = layers.Dense(256, activation="relu")(in_txt)
    x3 = layers.Dense(32, activation="relu")(in_cond) 
    
    merged = layers.Concatenate()([x1, x2, x3])
    h = layers.Dense(256, activation="relu")(merged)
    
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model([in_aud, in_txt, in_cond], [z_mean, z_log_var, z])
    
    # Decoder Inputs
    lat_in = keras.Input(shape=(latent_dim,))
    cond_in = keras.Input(shape=(cond_dim,))
    
    # Decoder Layers
    merged_dec = layers.Concatenate()([lat_in, cond_in])
    h_dec = layers.Dense(256, activation="relu")(merged_dec)
    
    d1 = layers.Reshape((26, 1))(layers.Dense(26, activation="sigmoid")(h_dec))
    d2 = layers.Dense(768)(h_dec)
    decoder = keras.Model([lat_in, cond_in], [d1, d2])
    
    return encoder, decoder