"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from TextCNN import create_model

np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "keras_data_set"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 10
num_epochs = 50

# Prepossessing parameters
sequence_length = 100
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

model_name_s = 'CNN-Better-Data'
model_load = False
grid_search = True


# ---------------------- Parameters end -----------------------


def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        from data_helpers import load_data
        x, y, vocabulary, vocabulary_inv_list = load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        y = y.argmax(axis=1)

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * 0.95)
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_test = x[train_len:]
        y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv


def perform_grid_search(x_train, y_train, x_test, y_test, vocabulary_inv):
    import sys
    old_stdout = sys.stdout

    from datetime import datetime
    log_file = open("grid_search" + model_name_s + ".log", "w")

    sys.stdout = log_file
    global model, batch_size
    model = KerasClassifier(build_fn=create_model,
                            x_train=x_train, x_test=x_test,
                            vocabulary_inv=vocabulary_inv,
                            model_type=model_type,
                            embedding_dim=embedding_dim,
                            min_word_count=min_word_count,
                            context=context,
                            sequence_length=sequence_length,
                            dropout_prob=dropout_prob,
                            filter_sizes=filter_sizes,
                            num_filters=num_filters,
                            hidden_dims=hidden_dims)
    # define the grid search parameters
    batch_size = [10, 20, 30]
    epochs = [30, 50]
    optimizer = ['SGD', 'Adagrad', 'Adam', 'Adamax']
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=10)
    # grid = GridSearch(model=model,num_threads=1)
    grid_result = grid.fit(x_train, y_train)
    print(grid_result)
    print('Test Score for Optimized Parameters:', grid.score(x_test, y_test))
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    sys.stdout = old_stdout
    log_file.close()


if __name__ == '__main__':
    # Data Preparation
    print("Load data...")
    x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source)

    if sequence_length != x_test.shape[1]:
        print("Adjusting sequence length for actual size")
        sequence_length = x_test.shape[1]

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    model = create_model(x_train=x_train, x_test=x_test,
                         vocabulary_inv=vocabulary_inv,
                         model_type=model_type,
                         embedding_dim=embedding_dim,
                         min_word_count=min_word_count,
                         context=context,
                         sequence_length=sequence_length,
                         dropout_prob=dropout_prob,
                         filter_sizes=filter_sizes,
                         num_filters=num_filters,
                         hidden_dims=hidden_dims)
    if model_load:
        model.load_weights(model_name_s)
    else:

        if grid_search:
            print("Grid Searching...")
            perform_grid_search(x_train=x_train, y_train=y_train,
                                x_test=x_test, y_test=y_test,
                                vocabulary_inv=vocabulary_inv)
        else:
            print("Training Model...")
            model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                      validation_data=(x_test, y_test), verbose=10)

    predictions = model.predict(x_test)
    i = 0
    y_pred = []
    for prediction in predictions:
        if prediction[0] >= 0.75:
            y_pred.append(1)
        else:
            y_pred.append(0)

    from sklearn.metrics import confusion_matrix

    y_true = y_test
    print(confusion_matrix(y_true, y_pred))
