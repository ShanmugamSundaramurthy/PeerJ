from keras.models import Model
from keras.optimizers.legacy import RMSprop
from keras.regularizers import l2
from spektral.layers import GCNConv
from spektral.utils import sp_matrix_to_sp_tensor


from keras import backend as K
from keras.layers import (
    Input,
    Dense,
    Conv1D,
    GRU,
    Lambda,
    Dot,
    Activation,
    Concatenate,
    Layer,
)
from keras.models import Model
from keras.optimizers.legacy import RMSprop


class Attention(Layer):
    def __init__(self, units=128, **kwargs):
        print("[INFO] Building Attention Layer")
        super(Attention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope("attention"):
            self.attention_score_vec = Dense(
                input_dim, use_bias=False, name="attention_score_vec"
            )
            self.h_t = Lambda(
                lambda x: x[:, -1, :],
                output_shape=(input_dim,),
                name="last_hidden_state",
            )
            self.attention_score = Dot(axes=[1, 2], name="attention_score")
            self.attention_weight = Activation("softmax", name="attention_weight")
            self.context_vector = Dot(axes=[1, 1], name="context_vector")
            self.attention_output = Concatenate(name="attention_output")
            self.attention_vector = Dense(
                self.units, use_bias=False, activation="tanh", name="attention_vector"
            )
            super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        return super(Attention, self).__call__(inputs, training, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        score_first_part = self.attention_score_vec(inputs)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"units": self.units})
        return config


def buildGCGRN(x_shape, adjacency_matrix, num_classes):
    print("[INFO] Building Graph Convolutional Gated Recurrent Network")
    x_in = Input(shape=x_shape)
    a_in = Input(tensor=sp_matrix_to_sp_tensor(adjacency_matrix))
    gcn = GCNConv(128, activation="relu", kernel_regularizer=l2(5e-4), use_bias=True)(
        [x_in, a_in]
    )
    cnn = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(gcn)
    gru = GRU(32, return_sequences=True)(cnn)
    attention = Attention(units=16)(gru)
    output = Dense(units=num_classes, activation="softmax")(attention)
    model = Model(inputs=[x_in, a_in], outputs=output)
    print("[INFO] Compiling Model Using RMSProp Optimizer")
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001 if num_classes == 2 else 0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model
