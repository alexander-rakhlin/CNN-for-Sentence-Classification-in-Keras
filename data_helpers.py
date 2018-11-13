import numpy as np
import re
import itertools
from collections import Counter

"""
Module of data helper functions.
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string: str) -> str:
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    Remove symbols such as brackets from string received in argument,
    and return new lowercase string.

    Parameters
    ----------
    string : str

    Returns
    -------
    string : str
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels() -> list:
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.

    Returns
    -------
    [x_text, y]

    x_text : list of str
        list of words. tokenized from text file.

    y : list of list
        labels for binary classification, [0, 1] and [1, 0].
        when text data labels are [pos, neg, pos, neg], label data
        y would be [[0, 1], [1, 0], [0, 1], [1, 0]]
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Tokenization
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences: list, padding_word="<PAD/>") -> list:
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences, in order to align matrix dimensions.


    Parameters
    ----------
    sentences : list of list
        [[sentence1], [sentence2], ..., [sentenceN]]
        Each sentence is list of words.

    padding_word : str
        Fill missed length of string with padding_word.
        When sequence length of longest sentence is 4, sentence=["Hello", "World"]
        would be padded_sentence=["Hello", "World", "<PAD/>", "<PAD/>"].

    Returns
    -------
    padded_sentences : list of str
        [[padded_sentence1], [padded_sentence2], ..., [padded_sentenceN]]
        All the sentences have same char string length.
    """
    sequence_length = max(len(sentence) for sentence in sentences)
    padded_sentences = []

    for sentence in sentences:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    return padded_sentences


def build_vocab(sentences: list) -> list:
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.

    Parameters
    ----------
    sentences : list of list
        [[sentence1], [sentence2], ..., [sentenceN]]
        Each sentence is list of words.

    Returns
    -------
    vocabulary : dict
        mapping from word to index (order of appearance in sentences).
        ex. {'<PAD/>': 0, 'Hello': 1, 'World': 2}

    vocabulary_inv : list
        list of words, sorted by word frequency in sentences.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences: list, labels: list, vocabulary: dict) -> list:
    """
    Maps sentences and labels to vectors based on a vocabulary.

    Parameters
    ----------
    sentences : list of list
        [[sentence1], [sentence2], ..., [sentenceN]]
        Each sentence is list of words.

    labels  : list
        list of labels.

    vocabulary : dict
        mapping from word to index (order of word frequency).
        ex. {'<PAD/>': 0, 'Hello': 1, 'World': 2}

    Returns
    -------
    x : numpy.ndarray
        2d array, tensor of word indices of sentences.

    y : numpy.ndarray
        1d array of label data.

    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)

    return [x, y]


def load_data() -> list:
    """
    Main function of this module.
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.

    Returns
    -------
    Returns [x, y, vocabulary, vocabulary_inv]

    x : numpy.ndarray
        2d array, tensor of word indices of sentences.

    y : numpy.ndarray
        1d array of label data.

    vocabulary : dict
        mapping from word to index (order of appearance in sentences).
        ex. {'<PAD/>': 0, 'Hello': 1, 'World': 2}

    vocabulary_inv : list
        list of words, sorted by word frequency in sentences.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data: list, batch_size: int, num_epochs: int) -> iterator:
    """
    Generates a batch iterator for a dataset.

    Parameters
    ----------
    data : list
    batch_size : int
    num_epochs : int
    Returns
    -------
    shuffled_data : iterator

    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]
