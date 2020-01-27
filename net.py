import tensorflow as tf


def make_model(hidden_arch_list, weight_norm_penalty, dropout_fraction):
    """Make tensorflow model based on config file.

    Let's model architecture be described in the config file without python
    knowledge.

    Args:
        hidden_arch_list (list(str)) : list in the format [(units, activation),...]
            defining the number of hidden units and activation function used in each
            hidden layer of the network. Activation choices include leaky_relu, relu, tanh,
            sigmoid, softmax, linear.
        weight_norm_penalty (float) : coefficient for L2 regularization. Reduces overfitting by
            constraining variables to stay close to the origin. Reccomended values are between
            0 and .1.
        dropout_fraction (float) : value >= 0 and < 1 that determines the percentage of weights
            randomly set to 0 on each pass during training. Reduces overfitting by mimicing large
            ensembles of submodels. Reccomended values are between 0 and .5.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((4,)))
    for layer_spec in hidden_arch_list:
        if layer_spec[1] != 'leaky_relu':
            activation = layer_spec[1]
            add_leaky = False
        else:
            activation = 'linear'
            add_leaky = True
        layer = tf.keras.layers.Dense(layer_spec[0], 
                                      activation=activation,
                                      kernel_regularizer=tf.keras.regularizers.l2(weight_norm_penalty),
                                      bias_regularizer=tf.keras.regularizers.l2(weight_norm_penalty),
                                    )
        model.add(layer)
        if dropout_fraction:
            model.add(tf.keras.layers.Dropout(dropout_fraction))
        if add_leaky: model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(3, 
                                    activation='linear',
                                    )
            )
    return model