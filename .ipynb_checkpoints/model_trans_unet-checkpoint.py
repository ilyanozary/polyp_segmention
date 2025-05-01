import tensorflow as tf
from tensorflow.keras import layers, models

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, emb_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = layers.Conv2D(emb_dim, kernel_size=patch_size, strides=patch_size)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.proj(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
        return self.norm(x)


class MLP(layers.Layer):
    def __init__(self, hidden_units, output_units, dropout_rate):
        super().__init__()
        self.fc1 = layers.Dense(hidden_units, activation='gelu')
        self.dropout = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(output_units)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(mlp_dim, embed_dim, dropout_rate)


    def call(self, x):
        x1 = self.norm1(x)
        attn_output = self.attn(x1, x1)
        x2 = self.dropout(attn_output) + x
        x3 = self.norm2(x2)
        return self.mlp(x3) + x2


class TransUNet(tf.keras.Model):
    def __init__(self, input_shape=(384, 384, 3), patch_size=16, emb_dim=768,
                 num_blocks=8, num_heads=8, mlp_dim=3072, dropout_rate=0.1):
        super().__init__()
        self.input_layer = layers.Input(shape=input_shape)
        self.encoder = PatchEmbedding(patch_size, emb_dim)
        self.transformer_blocks = [
            TransformerBlock(emb_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_blocks)
        ]
        self.reshape = layers.Reshape((input_shape[0] // patch_size, input_shape[1] // patch_size, emb_dim))

        self.decoder = models.Sequential([
            layers.UpSampling2D(size=(2, 2)), 
            layers.Conv2DTranspose(256, 3, strides=1, padding="same", activation='relu'),
            layers.UpSampling2D(size=(2, 2)),  
            layers.Conv2DTranspose(128, 3, strides=1, padding="same", activation='relu'),
            layers.UpSampling2D(size=(2, 2)),  
            layers.Conv2DTranspose(64, 3, strides=1, padding="same", activation='relu'),
            layers.UpSampling2D(size=(2, 2)),  
            layers.Conv2DTranspose(32, 3, strides=1, padding="same", activation='relu'),
            layers.Conv2D(1, kernel_size=1, activation='sigmoid')  
        ])



    def call(self, inputs):
        x = self.encoder(inputs)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.reshape(x)
        return self.decoder(x)


if __name__ == '__main__':
    model = TransUNet()
    dummy_input = tf.random.normal([1, 384, 384, 3])
    dummy_output = model(dummy_input)
    print("Output shape:", dummy_output.shape)