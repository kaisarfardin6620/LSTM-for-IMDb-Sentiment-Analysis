# LSTM-for-IMDb-Sentiment-Analysis


This project aims to perform sentiment analysis on IMDB movie reviews using deep learning techniques. The goal is to classify movie reviews as positive or negative based on their textual content.

## Dataset

The project utilizes the IMDB dataset, which consists of 50,000 movie reviews labeled as positive or negative. The dataset is split into 25,000 reviews for training and 25,000 reviews for testing.

## Methodology

1. **Data Preprocessing:**
   - Removing HTML tags, URLs, and special characters.
   - Converting text to lowercase.
   - Removing stop words and punctuation.
   - Lemmatizing words.

2. **Feature Extraction:**
   - Tokenizing the text using the Keras Tokenizer.
   - Padding sequences to a fixed length.

3. **Model Building:**
   - Using a sequential model with an embedding layer, LSTM layer, and dense layers.
   - Applying dropout and batch normalization for regularization.

4. **Model Training:**
   - Training the model using the Adam optimizer and binary cross-entropy loss.
   - Implementing early stopping to prevent overfitting.

5. **Model Evaluation:**
   - Evaluating the model's performance using accuracy, precision, recall, F1-score, and AUC score.

## Results

The model achieved an accuracy of approximately 85% on the test set.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NLTK
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Wordcloud

## Usage

1. Clone the repository.
2. Install the required packages.
3. Run the Jupyter notebook `imdb_sentiment_analysis.ipynb`.

## Conclusion

This project demonstrates the effectiveness of deep learning models for sentiment analysis. The model can be further improved by exploring different architectures, hyperparameters, and preprocessing techniques.
