This NLP project involves sentiment analysis on the rtmr and s140 datasets, each requiring preprocessing due to discrepancies in class balance and formats.

Data preparation steps included sample size limitation, label adjustment, tokenization, stop-word removal, splitting hyphen-separated words, and lowercase conversion.

Two text representation methods were applied: a custom tokenization scheme for Naive Bayes classifier and Tensorflow's tokenization for neural networks.

Machine learning architectures utilized included a Naive Bayes algorithm, Recurrent Neural Networks (RNNs), Gated Recurrent Units (GRUs), and Convolutional Neural Networks (CNNs).

Model performance showed positive results with some limitations, notably struggling with neutral and negative sentiments. F1 precision and recall improved to ~60% for the s140 dataset using thresholding techniques. Accuracy reached 77% and 75% respectively for the RNN and CNN models.
