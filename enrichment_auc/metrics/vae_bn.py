from keras import backend as K
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Lambda, ReLU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.utils import plot_model
from tensorflow.random import set_seed


class VAE_BN(object):
    def __init__(self, nSpecFeatures, intermediate_dim, latent_dim, seed):
        self.nSpecFeatures = nSpecFeatures
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.seed = seed

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(
            shape=(batch, dim), seed=self.seed
        )  # random_normal (mean=0 and std=1)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def get_architecture(self, verbose=False, lr=1e-3):
        set_seed(self.seed)
        # =========== 1. Encoder Model================
        input_shape = (self.nSpecFeatures,)
        inputs = Input(shape=input_shape, name="encoder_input")

        try:
            dims = tuple(self.intermediate_dim)
        except TypeError:  # not iterable
            dims = (self.intermediate_dim,)

        h = inputs

        for dim in dims:
            h = Dense(dim, kernel_initializer=GlorotUniform(seed=self.seed))(h)
            h = BatchNormalization()(h)
            h = ReLU()(h)

        z_mean = Dense(
            self.latent_dim,
            name="z_mean",
            kernel_initializer=GlorotUniform(seed=self.seed),
        )(h)
        z_mean = BatchNormalization()(z_mean)
        z_log_var = Dense(
            self.latent_dim,
            name="z_log_var",
            kernel_initializer=GlorotUniform(seed=self.seed),
        )(h)
        z_log_var = BatchNormalization()(z_log_var)

        # Reparametrization Trick:
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")(
            [z_mean, z_log_var]
        )
        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        if verbose == "auto" or verbose > 1:
            print("==== Encoder Architecture...")
            encoder.summary()
        # plot_model(encoder, to_file="VAE_BN_encoder.png", show_shapes=True)

        # =========== 2. Decoder Model================
        latent_inputs = Input(shape=(self.latent_dim,), name="Latent_Space")

        hdec = latent_inputs

        for dim in dims[::-1]:
            hdec = Dense(dim, kernel_initializer=GlorotUniform(seed=self.seed))(hdec)
            hdec = BatchNormalization()(hdec)
            hdec = ReLU()(hdec)

        outputs = Dense(
            self.nSpecFeatures,
            activation="sigmoid",
            kernel_initializer=GlorotUniform(seed=self.seed),
        )(hdec)
        decoder = Model(latent_inputs, outputs, name="decoder")
        if verbose == "auto" or verbose > 1:
            print("==== Decoder Architecture...")
            decoder.summary()
        # plot_model(decoder, to_file='VAE_BN__decoder.png', show_shapes=True)

        # =========== VAE_BN: Encoder_Decoder ================
        outputs = decoder(encoder(inputs)[2])
        VAE_BN_model = Model(inputs, outputs, name="VAE_BN")

        # ====== Cost Function (Variational Lower Bound)  ==============
        "KL-div (regularizes encoder) and reconstruction loss (of the decoder): see equation(3) in our paper"
        # 1. KL-Divergence:
        kl_Loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_Loss = K.sum(kl_Loss, axis=-1)
        kl_Loss *= -0.5
        # 2. Reconstruction Loss
        reconstruction_loss = categorical_crossentropy(
            inputs, outputs
        )  # Use sigmoid at output layer
        reconstruction_loss *= self.nSpecFeatures

        # ========== Compile VAE_BN model ===========
        model_Loss = K.mean(reconstruction_loss + kl_Loss)
        VAE_BN_model.add_loss(model_Loss)
        VAE_BN_model.compile(optimizer=Adam(learning_rate=lr))
        return VAE_BN_model, encoder
