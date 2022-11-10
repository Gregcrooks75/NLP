# NLP
NLP project
##1. Critical discussion about the datasets

There is a discrepancy in terms of classes with the rtmr test and training datasets having 5
sentiment classification of 0,1,2,3, and 4, the s140 training set having only 2 sentiment
classifications (0 and 4), and the s140 testing set having 3 sentiment classifications (0,2,4).
Furthermore, there is a difference in terms of format. Given the format of the sentences, it is
essential to implement pre-processing in order to analyze the data accordingly and reduce the
volume of data by only selecting the most insightful information. Both datasets are huge,
containing millions of data points. There is additional concern for computational costs given that
the s140 train dataset has 1599999 observations. The s140 train dataset was found to have
balance between class 0 and class 4 which is desirable. However, the rtmr dataset was found to
have a lot of imbalance, which <5% class 0, ~20% class 1, ~50% class 2, ~20% class 3, and
~6% class 4. This imbalance requires fixing through sampling or weighting so machine learning
models do not predict class 2 too often.

##2. Description and justification of the data preparation steps used.

We first and foremost chose to limit the maximum number of observations used to 5000, which
was applicable to both training sets. However, one issue arising was that the s140 training set
had data sorted such that the first half was a 0 classifications and the second half was 4
classifications. Consequently, to ensure that data was random, we selected one out of every
200 observations for the s140 training set, and one out of 30 observations for the rtmr training
set.
To initially address the issue of labels, we decided to convert the 1s to 0s and 4s to 1s in the
s140 dataset for training. Furthermore, adding the same headers for every dataset enabled us
to carry out our preprocessing more effectively by specifically converting the “Phrase” columns
to a list. This enabled us to tokenize words. Subsequently, we removed stop words for a more
relevant analysis, separated words with hyphens into 2 words, changed every letter to
lowercase, and removed every symbol which was in the punctuation list. Finally, we stored the
processed words and labels inside different variables to use again.

##3. Description and justification of the text representation method(s) used.

A self written tokenisation scheme for the Naive Bayes classifier, and using Tensorflow’s
tokenization for the neural network part. For the Naive Bayes classifier, a count vectorizer was
used to convert the sentences into a vector array suitable for machine learning. The count
vectorises assigns a column to the N most common words in the training set. Each column
represents the count of that word in a given review. These vectors were not normalised as
Naive Bayes does not require normalised data.
For the neural networks, each word was assigned an ID. An embedding layer was put as the
first layer in the model. The embedding layer learns to convert each ID into a vector of a
specified length. Therefore each review is converted into a sequence of vectors. These vectors
needed to be padded to the max length of the sentence in the training set. As these vectors
form part of the network, they will be passed onto the next layer during training and inference.

##4. Description and commentary on the machine learning architectures used.

Naive Bayes algorithm calculates the probability of a given output class using the prior,
likelihood and evidence found in the data. It does this for each feature and each class and
combines the predicted probabilities by assuming all features are independent (hence the Naive
name). This is suitable for sentiment analysis as it is likely that the model could learn that certain
words indicate a good sentiment i.e. ‘good’/’amazing’, and other words could indicate a bad
sentiment ‘rubbish’/’boring’.
Recurrent neural networks are networks which have a layer which accepts an input and a state,
and outputs an output and a state. This allows the layer to form a memory of what has come
previously, and model long range dependencies. For text this is important as the words at the
start of a sentence may be very relevant at the end. Gated Recurrent Units (used later) and
LSTMs further allow this long range modeling through selectively remembering certain pieces of
information and forgetting others. Convolutional neural networks enable weight sharing by
moving a filter over the input to make a feature map.

##5. Detailed performance evaluation of the developed machine learning models.

By analyzing the confusion matrix, we can see that the model has learnt some signal. It predicts
1 when the true label is 1. However the model tends to also predict 1 when the true label is 0
and 2. To fix this, some appropriate oversampling and undersampling will need to be done.
Training on the RTMR and testing on the s140 shows a similar story. The model predicts class 1
far too often. However, when it does select class 0 and 1 it does so with good precision, again
showing the model has learnt some signal.


The train on the s140 and test on the s140 is the best result set. There is a notable line down
the diagonal which is desired for a confusion matrix, and the F1 precision and recall are all
~60%. This was achieved by having the model output a number between 0 and 1 and setting a
threshold as to whether a review is neutral. In other words, if the model output is between 0.33
and 0.66, the review is neutral. These thresholds were set arbitrarily. This result is very
interesting as it shows the model has not only learnt the sentiments of good and bad, but only
indicates when the review is more ambiguous, and this indication matching the neutral reviews
from the testing set.

The train s140, test rtmr doesn’t appear to perform very well. This could be for a number of
reasons, firstly the threshold trick was used again to output a neutral class. However, the RTMR
dataset in its raw state has 5 classes, and the labels of 1 were put mean bad (0), and labels of 3
grouped with 4, which then became 2 (as 1 is neutral). This will mean the extremes on the rtmr
will be slightly closer to normal than they would normally be. This means the s140 may predict
these data points to be neutral (and their true sentiments may be close to neutral), when in fact
they are labeled as extreme bad or good. Looking at the confusion matrix, this appears to be
exactly what has happened.

An RNN and CNN were trained for the S140 dataset. Here validation s140 was used during
training to track progress and overfitting. The validation set had the neutral labels removed, to
enable tensorflow to process it and calculate an accuracy. We can see from the training that the
accuracy reaches 77%, which is a very good score. This image is the training of the RNN,
however the CNN is very similar with 75%.

For the confusion matrix, neutral reviews were classified by using a threshold on the output
confidence of the model. For both the RNN and CNN we can see the model does very well on
the true labels 0 and 2, but struggles with the true labels of 1. It performs worse on these labels
than the naive bayes. This is likely to be because the neural networks learn exact sequences to
signify good or bad, as opposed to outputting a probability based on word frequencies.
The 1d CNN and RNN both have similar scores. However the RNN does slightly better. This is
expected as the RNN can model long range dependencies which are essential to text better
than a 1d cnn. Note that these results could likely be improved by using the full dataset and
using a bigger network. Neural networks are extremely data hungry so will likely improve with a
large increase in data such as 200 times the amount.

The CNN and RNN were not found to be significantly different for this task. We can see that a
predicted label of 2 is made too often, even in the case of the true label being 1 and 3. We can
only see that the most common true label in the top row is 1, and the most common true label in
the bottom row 3. This means the model is predicting extremes too often when there is only a
mild sentiment there which could be from an imbalanced dataset. To fix this issue, it would be
proposed to use more data.

##6. Critical discussion on the achieved results, including potential limitations and usage
instructions/suggestions.

For the naive bayes models, words need to be tokenized in the same manner that was done in
the data preprocessing. It is very important that each word corresponds to the same position in
the vector when the words are counted. A vector created this way would then be suitable for the
model. This means that for an app or demo to be created more than just the model needs to be
saved, the whole pipeline needs to be saved.
The same concept holds for the neural network models. In this case the same tokenizer model
from tensorflow would need to be used, and the same padding length. After this preprocessing
could the string then be fed to the model.
This means that building a usable demo would be slightly easier for the neural network model,
as the Python script would only need to load the tokenizer class from tensorflow and the model
file.
