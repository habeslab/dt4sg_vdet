# generator.py
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, regularizers #type: ignore

def build_generator(latent_dim: int,
                    output_dim: int,
                    hidden_dims=(256, 256, 128),
                    l2: float = 1e-5) -> tf.keras.Model:
    """
    Generatore per dati tabellari (N, F). Output: vettore in R^F.
    """
    z = layers.Input(shape=(latent_dim,), name="z")
    x = z
    for i, h in enumerate(hidden_dims):
        x = layers.Dense(h,
                         activation="gelu",
                         kernel_regularizer=regularizers.l2(l2),
                         name=f"g_dense_{i}")(x)
    out = layers.Dense(output_dim, activation="tanh", name="g_out")(x)
    return tf.keras.Model(z, out, name="Generator")
