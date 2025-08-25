# discriminator.py
from __future__ import annotations
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, regularizers  #type: ignore

class MinibatchStdDev(layers.Layer):
    """
    Implementazione semplice per dati tabellari (N, F):
    calcola la std per-feature sul batch, ne fa la media (scalare),
    e la concatena come feature addizionale.
    """
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=None):
        # inputs: [B, F]
        x = tf.cast(inputs, tf.float32)
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        var  = tf.reduce_mean(tf.square(x - mean), axis=0, keepdims=True)
        std  = tf.sqrt(var + self.epsilon)              # [1, F]
        avg_std = tf.reduce_mean(std, axis=1, keepdims=True)  # [1, 1]
        # broadcast a [B, 1]
        b = tf.shape(x)[0]
        mb_feat = tf.tile(avg_std, [b, 1])
        return tf.concat([x, mb_feat], axis=-1)         # [B, F+1]

def build_discriminator(input_dim: int,
                        hidden_dims=(256, 128, 64),
                        dropout: float = 0.2,
                        l2: float = 1e-5,
                        use_mbstd: bool = True) -> tf.keras.Model:
    """
    Discriminatore 3-classi: [0=classe A, 1=classe B, 2=falso]
    """
    inp = layers.Input(shape=(input_dim,), name="disc_input")
    x = inp
    if use_mbstd:
        x = MinibatchStdDev(name="mbstd")(x)

    for i, h in enumerate(hidden_dims):
        x = layers.Dense(h,
                         activation="leaky_relu",
                         kernel_regularizer=regularizers.l2(l2),
                         name=f"dense_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    logits = layers.Dense(3, activation=None, name="logits")(x)
    return tf.keras.Model(inp, logits, name="Discriminator")
