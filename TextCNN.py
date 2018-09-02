import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dropout, Convolution1D, MaxPooling1D, Flatten, Concatenate, Dense

from w2v import train_word2vec


def create_model(x_train, x_test, vocabulary_inv, model_type, embedding_dim, min_word_count,context, sequence_length, dropout_prob,filter_sizes,num_filters,hidden_dims,optimizer="Adagrad"):
    # Prepare embedding layer weights and convert inputs for static model
    print("Model type is", model_type)
    if model_type in ["CNN-non-static", "CNN-static"]:
        embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                           min_word_count=min_word_count, context=context)
        if model_type == "CNN-static":
            x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
            x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
            print("x_train static shape:", x_train.shape)
            print("x_test static shape:", x_test.shape)

    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    # Build model
    if model_type == "CNN-static":
        input_shape = (sequence_length, embedding_dim)
    else:
        input_shape = (sequence_length,)
    model_input = Input(shape=input_shape)
    # Static model does not have embedding layer
    if model_type == "CNN-static":
        z = model_input
    else:
        z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
    z = Dropout(dropout_prob[0])(z)
    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)
    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    # Initialize weights with word2vec
    if model_type == "CNN-non-static":
        weights = np.array([v for v in embedding_weights.values()])
        print("Initializing embedding layer with word2vec weights, shape", weights.shape)
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights([weights])
    return model
