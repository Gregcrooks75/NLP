# Natural Language Processing (NLP) Project

This project focuses on sentiment analysis using various datasets and machine learning models to classify text data based on sentiment.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sentiment analysis is a crucial aspect of Natural Language Processing (NLP) that involves determining the sentiment expressed in textual data. This project explores sentiment analysis using different datasets and machine learning models, including Naive Bayes, Recurrent Neural Networks (RNNs), Gated Recurrent Units (GRUs), and Convolutional Neural Networks (CNNs).

## Project Structure

The repository comprises the following files:

- `NLP.ipynb`: A Jupyter Notebook containing the data preprocessing steps, model training, evaluation, and analysis.
- `README.md`: This file, providing an overview of the project.

## Data Sources

The project utilizes the following datasets:

- **RTMR Dataset**: Contains movie reviews with sentiment labels ranging from 0 to 4.
- **S140 Dataset**: Comprises tweets labeled with sentiments, with variations in sentiment classes between training and testing sets.

These datasets are used to train and evaluate the sentiment analysis models.

## Data Preprocessing

The collected data underwent the following preprocessing steps:

- **Cleaning**: Removal of HTML tags, special characters, and irrelevant information.
- **Tokenization**: Breaking down text into individual words or tokens.
- **Lemmatization**: Reducing words to their base or root form.
- **Stopword Removal**: Eliminating common words that do not contribute to sentiment (e.g., "and", "the", "is").

These steps are detailed in the `NLP.ipynb` notebook.

## Modeling

The following machine learning models were implemented:

- **Naive Bayes Classifier**: Utilizes probabilistic methods for classification.
- **Recurrent Neural Networks (RNNs)**: Captures sequential dependencies in text data.
- **Gated Recurrent Units (GRUs)**: An advanced version of RNNs that mitigates the vanishing gradient problem.
- **Convolutional Neural Networks (CNNs)**: Extracts local features from text data.

Each model was trained and evaluated on the preprocessed datasets, with performance metrics recorded for comparison.

## Results

The analysis revealed the following insights:

- **Naive Bayes Classifier**: Achieved moderate accuracy but struggled with neutral sentiments.
- **RNN and GRU Models**: Demonstrated improved performance, capturing sequential patterns effectively.
- **CNN Model**: Showed competitive results, highlighting its capability in text classification tasks.

Detailed results and visualizations are available in the notebook.

## Usage

To replicate the analysis:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Gregcrooks75/NLP.git
