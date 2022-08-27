
# Applying Text Classification with Convolutional Neural Network Architecture to Detect Fake News

# Imports
import pandas as pd
import numpy as np
import os

from wordcloud import WordCloud

from tensorflow.python import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt


# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('This Computation is running on {}'.format(device))


# Loading fake news data set
# Source: https://www.kaggle.com/datasets/ruchi798/source-based-news-classification

columns = [
    #    "author",
    #    "published",
    #    "title",
    #    "text",
    "language",
    #    "site_url",
    #    "main_img_url",
    #   "type",
    "label",
    "title_without_stopwords",
    "text_without_stopwords",
]

df = pd.read_csv(os.path.join('data', 'news_articles.csv'), usecols=columns)
print(df.head())


# Data Pre-processing

# Deleting news articles in German from data set (N=72)
print(f'Shape of the dataset: {df.shape}')
df = df[df.language != 'german']
df = df.drop(['language'], axis=1)
print(f'Shape of the dataset: {df.shape}')

# Checking number of valid data entries
print(df.describe())
print(df.info())

# Removing NaN values
df.isnull().sum()
df.dropna(inplace=True)
print(f'Shape of the dataset: {df.shape}')

# Converting string to numeric values
print(df.label.value_counts())
df["label"] = df["label"].replace({'Real': 0, 'Fake': 1})

# Checking distribution of target variable
print(df.label.value_counts())
print(df.label.value_counts() / len(df))

print(df.head(2))
print(f'Shape of the dataset: {df.shape}')


# Description of input data: Visualising frequency of words using WordCloud package
wc = WordCloud(background_color="white", max_words=200,
               max_font_size=256,
               random_state=42, width=1000, height=1000)
wc.generate(' '.join(df['title_without_stopwords'] +
            ' ' + df['text_without_stopwords']))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()


# Defining target and feature
y = np.array(df['label'])
# Texts are appended to the titles and processed together
print('text_without_stopwords'[0])
print('title_without_stopwords'[0])
x = np.array(df['title_without_stopwords'] +
             ' ' + df['text_without_stopwords'])
print(x[0])


# Pre-processing of text data: Mapping text into vectors
'''
    Pre-Processing of text data includes
    - Tokenization
    - Word indexing
    - Vectorization
    - Padding
'''

# Set parameters for tokenization and embedding
max_numwords = 20000
pad_len = 1000
embedding_dim = 300
batch_size = 128

# Tokenizing text
'''
    Breaking up the original raw text into component pieces (tokens) to allow for further Natural
    Language Processing (NLP) operations
    see https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

    -  Define the maximum number of words to keep, based on word frequency. Only the most common
       num_words-1 words will be kept
    -  Update internal vocabulary based on a list of texts:
       tokenizer.fit_on_texts creates the vocabulary index based on word frequency, that is,
       an index dictionary in which every word gets a unique integer value, with lower (higher)
       integers indicating words that occur more frequently (rarely)
'''
tokenizer = Tokenizer(num_words=max_numwords)
tokenizer.fit_on_texts(x)

# Word indexing: Map orginal word to number
'''
    Transforms each text in texts to a sequence of integers. Specifically, each word in
    the text is replaced with its corresponding integer value from the word_index dictionary.
    - Required before using texts_to_sequences
    - Mapping is preserved in word_index property of tokenizer
'''
word_index = tokenizer.word_index
print('Found {} unique tokens and {} lines '.format(len(word_index), len(x)))
# 42,915 unique tokens and 1,973 lines

# Vectorization
''' Transforms each text in x to a sequence of integers
    - Only top `num_words-1` most frequent words will be taken into account
    - Only words known by the tokenizer will be taken into account
    ----
    Args: A list of texts (strings)
    Returns: A list of sequences
    ----
'''
sequences = tokenizer.texts_to_sequences(x)
print("Sequence: ", sequences[0])

# Padding
'''
    Standardization of sequence length:
    Set maximum length of all sequences to 1,000, that is, add padding (index 0) to news with less
    than 1,000 words and truncating to long ones
    see https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
'''
x = pad_sequences(sequences, maxlen=pad_len)
print("Padding Result: ", x[0])


# Splitting data set into test and train data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=45)

print('Shape of training data tensor:', x_train.shape)
print('Length of training label vector:', len(y_train))
print('Shape of validation data tensor:', x_test.shape)
print('Length of validation label vector:', len(y_test))


# Creating PyTorch DataLoaders for training and test data sets
'''
    Takes as input a single batch of data which has a list of text documents (batch_size=128) and
    their respective target labels; final features array is a 2D array, with as many rows as
    there are articles, and as many columns as the specified sequence length (seq_length).
'''

print('Train: ', end="")
train_dataset = TensorDataset(torch.LongTensor(x_train),
                              torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
print(len(train_dataset), 'messages')

print('Test: ', end="")
test_dataset = TensorDataset(torch.LongTensor(x_test),
                             torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)
print(len(test_dataset), 'messages')

print('Shape of training data tensor:', x_train.shape)
print('Length of training label vector:', len(y_train))
print('Shape of validation data tensor:', x_test.shape)
print('Length of validation label vector:', len(y_test))


# Assignment of Word Embeddings
'''
    In an embedding, words are represented by dense vectors. The position of a word within the
    vector space is learned from text and is based on the words that surround the word when it is
    used. The present work applies the pre-trained word embedding GloVe; GloVe method is built on
    the idea to derive semantic relationships between words based on the word-word
    co-occurrence matrix; embeddings are provided for various dimensions and include 400,000 english
    words

    Source: https://nlp.stanford.edu/projects/glove/

    At baseline, an embedding dimension of 100 is applied; increasing the embedding dimension
    results in higher accuracy without significantly increasing training time. The final model is
    thus trained with an embedding dimension of 300
'''
# Set path to pre-trained word embedding file GloVe, applying a 300-dimensional embedding
path_emb = os.path.join('data', 'glove.6B.300d.txt')

# Prepare embedding index including 400,000 word vectors from pre-trained embedding file
embeddings_index = {}
with open(path_emb, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# Prepare embedding matrix
'''
   Embedding layer has shape (N, d) with N the size of the vocabulary and d the embedding dimension
'''
embedding_matrix = np.zeros((max_numwords, embedding_dim))
n_not_found = 0
for word, i in word_index.items():
    if i >= max_numwords-2:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Set words not found in embedding index to zero
        embedding_matrix[i+2] = embedding_vector
    else:
        n_not_found += 1

embedding_matrix = torch.FloatTensor(embedding_matrix)
print('Shape of embedding matrix:', embedding_matrix.shape)
print('Words not found in embeddings:', n_not_found)


# Set up Network

class Net(nn.Module):
    def __init__(self):
        '''
        Initialization of neural network architecture, specifying amount of layers and neurons:
        - Input is a batch of news articles. These go through a pre-trained embedding layer.
          -> Note that input x is already in embedding vectors from GloVe
        - Embedding layer: Pre-trained word-embeddings with N the size of the vocabulary (20,000)
          and d the embedding dimension (300), i.e., the number of word features
        - Three one-dimensional CNN layers for extraction of local features by using 128 filters of
          size 5 (5 being the number of sequential groups of words to look at); these go through a
          ReLu activation
        - Feature vectors generated by CNN are pooled by feeding them into a 1D MaxPooling layer
        - The maximum values from the convolutional and pooling layers are concatenated and
          passed to a fully-connected linear layer and output layer
        '''
        super(Net, self).__init__()
        self.embed = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True)
        # (300, 128, kernel_size=(5,), stride=(1,) -> default stride
        self.conv1 = nn.Conv1d(300, 128, 5)
        self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(128, 128, 5)
        self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(128, 128, 5)
        self.pool3 = nn.MaxPool1d(35)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 2)

    # Combining Layers
    def forward(self, x):
        '''
        Defines how a batch of inputs, x, passes through the model layers;
        Feeding input x through layers and activation functions and returning output y after
        propagating through the network
        '''
        x = self.embed(x)
        # input: [128, 1, 300, 1000]  -> Swaps 2nd and 3rd dimension
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # the size -1 is inferred from other dimensions
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# Initialize training

model = Net().to(device)

# Initialize optimizer with model parameters and learning rate.
optimizer = torch.optim.Adam(model.parameters(), 0.009)
# Initialize loss function
criterion = nn.CrossEntropyLoss()

# Print model summary
print(model)


# Function for training inizilization
'''
    Create function for training neural network:
    - Takes model, loss function, optimizer, train data loader, validation data
      loader, and number of epochs as input
    - Executes training for n epochs; for each epoch, the function loops through training data
      in batches using the train data loader, which returns vectorized data and their labels for
      each batch
    - For each batch, a forward pass-through network is performed to make predictions,
      calculate loss (using predictions and actual target labels), calculate gradients, and
      update network parameters.
    - Records loss for each batch and prints the average training loss at the end of each epoch
'''

epoch_loss_vis = []


def train(epoch):
    # Set model to training mode
    model.train()
    epoch_loss = 0.
    batch_loss = []

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):

        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Set all gradients to zero since it is a new iteration and optimization round
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss based on error regarding prediction and target value
        loss = criterion(output, target)
        epoch_loss += loss.data.item()
        batch_loss.append(loss.item())

        # Compute gradients for all parameters
        loss.backward()

        # Update all parameters using the gradients and optimizer formula
        optimizer.step()

    # Store loss for later visualization
    epoch_loss_vis.append(np.mean(batch_loss))
    epoch_loss /= len(train_loader)
    print('Train Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))


# Start training

epochs = 100

for epoch in range(1, epochs + 1):
    train(epoch)

# Plot training result
plt.plot(epoch_loss_vis)
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.title('Loss over Epochs: Emb_dim=300, batch=128, kernel=5')
plt.show()


# Check accuracy on test data set
'''
    Create function that takes input model, loss function, and validation data loader to
    calculate validation loss and accuracy
'''
correct = 0
total = 0
pred_vector = torch.LongTensor()

y_pred = []
y_true = []

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        outputs = model(inputs.cpu())

        _, predicted = torch.max(outputs, 1)
        pred_vector = torch.cat((pred_vector, predicted))

        total += labels.size(0)
        correct += (predicted == labels.cpu()).sum().item()

        # Save Prediction
        output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)

        # Save true values
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

        accuracy = 100. * correct / len(test_loader.dataset)

print("The Neural Network accuracy is at {}%".format(round(accuracy)))


def evaluation_measures(y_true, y_pred):
    '''
    Evaluate model predictions y_pred in view of true labels y_test
    Returns
    - cr : dict. classification report for the input
    - fpr, tpr, roc_auc : False-positive-rate, true-positive-rate and area-under-curve
    - cm : Confusion matrix
    '''

    # Classification report.
    cr = metrics.classification_report(y_true, y_pred)

    # ROC curve.
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # Confusion matrix.
    cm = metrics.confusion_matrix(y_true, y_pred)

    return cr, (fpr, tpr, roc_auc), cm


def plot_roc(auc_tuple):
    '''
       Plots the ROC curve for input data
       Includes fpr, tpr and auc (area-under-curve).
    '''
    fpr, tpr, roc_auc = auc_tuple
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def plot_cm(cm):
    '''
       Plots the confusion matrix for input data
       Includes True Negatives, False Positives, False Negatives, and True Positives
    '''
    tn, fp, fn, tp = cm.ravel()
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Reds")
    plt.show()


# Evaluate model using predictions
cr, roc, cm = evaluation_measures(y_true, y_pred)
print(cr)
plot_roc(roc)
plot_cm(cm)


# Save neural network
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss_vis,
}, "export.pt")
