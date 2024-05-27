import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load training data
train_data = pd.read_csv('Training_memes_dateset_test.csv')

# Create a pipeline that combines a text vectorizer with a Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(train_data['text'], train_data['is_ironic'])

# Save the model to a file
joblib.dump(model, 'irony_classifier_model.pkl')

# Function to predict irony in new texts
def predict_irony(text):
    model = joblib.load('irony_classifier_model.pkl')
    prediction = model.predict([text])
    return 'Ironic' if prediction[0] == 1 else 'Not Ironic'

# Test the model with new data
test_data = pd.read_csv('Testing_memes_dataset.csv')
test_data['prediction'] = test_data['text'].apply(predict_irony)

print(test_data)
