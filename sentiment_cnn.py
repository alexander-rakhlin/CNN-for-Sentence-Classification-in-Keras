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

np.random.seed(1234)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from TextCNN import create_model

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8, 12, 2)
num_filters = 25
dropout_prob = (0.5, 0.9)
hidden_dims = 50

# Training parameters
batch_size = 10
num_epochs = 40

# Preproceessing parameters
sequence_length = 100
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 10
context = 5

model_name_s = 'CNN-'
model_load = False
grid_search = True

# ---------------------- Parameters end -----------------------


def plot_model(model, filename='model.png'):
    from keras.utils import plot_model
    plot_model(model, to_file=filename)



def load_data():
    from data_helpers import load_data
    x, y, vocabulary, vocabulary_inv_list = load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.95)
    print("Train Length:", train_len)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv


def perform_grid_search(x_train, y_train, x_test, y_test, vocabulary_inv):
    import sys
    old_stdout = sys.stdout

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
    batch_size = [10]
    epochs = [10, 15, 20, 25, 30, 35]
    optimizer = ['Adamax']
    # batch_size = [10, 20]
    # epochs = [50, 75, 100]
    # optimizer = ['Adagrad', 'Adam', 'Adamax']
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
    x_train, y_train, x_test, y_test, vocabulary_inv = load_data()

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
            model.fit(x_train, y_train, batch_size=batch_size,
                      epochs=num_epochs, shuffle=True, verbose=2)
            model.save(model_name_s)

            score, acc = model.evaluate(x_test, y_test,
                                        batch_size=batch_size)
            print('Test score:', score)
            print('Test accuracy:', acc)
            plot_model(model)
