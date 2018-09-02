# Convolutional Neural Networks for Sentence Classification

Train convolutional network for sentiment analysis. Based on "Convolutional Neural Networks for Sentence Classification" by Yoon Kim, [link](http://arxiv.org/pdf/1408.5882v2.pdf). Inspired by Denny Britz article "Implementing a CNN for Text Classification in TensorFlow", [link](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85%

## Some difference from original article:
* larger IMDB corpus, longer sentences; sentence length is very important, just like data size
* smaller embedding dimension, 20 instead of 300
* 2 filter sizes instead of original 3
* much fewer filters; experiments show that 3-10 is enough; original work uses 100
* random initialization is no worse than word2vec init on IMDB corpus
* sliding Max Pooling instead of original Global Pooling

## Dependencies

* The [Keras](http://keras.io/) Deep Learning library and most recent [Theano](http://deeplearning.net/software/theano/install.html#install) backend should be installed. You can use pip for that. 
Not tested with TensorFlow, but should work.

without spell correction
Test score: 0.17737579543441062
Test accuracy: 0.9088447583496355

Test score: 0.298266075363228
Test accuracy: 0.9043321224326261

Test score: 0.1871837218492107
Test accuracy: 0.9124548669756535

smaller context
Test score: 0.18849021942516123
Test accuracy: 0.9196750840125101


Epoch 24 - 0.93

batch_size=10, epochs=20, optimizer=Adamax, score=0.9218482526281665, total= 6.5min
