import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('Indian-Name.csv', encoding='latin-1')

# Convert names to lowercase
data['Name'] = data['Name'].str.lower()

# Vectorize the names
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,3))
X = vectorizer.fit_transform(data['Name'])

# Define the target labels (male=0, female=1)
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SGD model with hinge loss (SVM-like) and 10 epochs
model = SGDClassifier(loss='hinge', max_iter=10, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))

# Function to predict gender for new names
def predict_gender(names):
    X_new = vectorizer.transform(names)
    predictions = model.predict(X_new)
    return predictions

# Test the prediction function
custom_names = ["name1", "name2", "name3", "name4"]
predicted_genders = predict_gender(custom_names)

for name, gender in zip(custom_names, predicted_genders):
    gender_label = 'Male' if gender == 0 else 'Female'
    print(f"{name}: Predicted gender - {gender_label}")



import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# List of common abbreviations
ABBREVIATIONS = {'Dr.', 'Mr.', 'Mrs.', 'Inc.', 'U.S.', 'U.S.A.', 'Jr.', 'Sr.'}

# Function to extract features from a word and its context
def extract_features(word, prev_word, next_word):
    features = {
        'ends_with_period': word.endswith('.'),
        'ends_with_exclamation': word.endswith('!'),
        'ends_with_question': word.endswith('?'),
        'is_numeric': bool(re.search(r'\d', word)),
        'is_abbreviation': word in ABBREVIATIONS,
        'next_word_capitalized': next_word[0].isupper() if next_word else False,
        'prev_word_lowercase': prev_word.islower() if prev_word else False,
        'word_length': len(word),
    }
    return features

# Prepare data (you would normally load this from a file)
data = [
    ("This is a sentence.", True),
    ("Mr. Smith went to Washington.", False),
    ("The price is $9.99.", True),
    ("What time is it?", True),
    ("Wow!", True),
    ("The company is called Acme, Inc.", False),
    ("He scored 4.5 on the test.", False),
    ("Is this the end?", True),
    ("No, it isn't.", True),
    ("Dr. Johnson is here.", False),
    ("The temperature is 98.6 degrees.", False),
    ("This is amazing!", True),
]

# Prepare features and labels
X = []
y = []

for sentence, label in data:
    words = sentence.split()
    for i, word in enumerate(words):
        prev_word = words[i-1] if i > 0 else ""
        next_word = words[i+1] if i < len(words) - 1 else ""
        features = extract_features(word, prev_word, next_word)
        X.append(features)
        y.append(label if i == len(words) - 1 else False)  # Label only the last word in a sentence as True

# Convert to DataFrame for easier handling
df = pd.DataFrame(X)
df['is_end_of_sentence'] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df.drop('is_end_of_sentence', axis=1), df['is_end_of_sentence'], test_size=0.2, random_state=42)

# Train the decision tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict if a word is at the end of a sentence
def predict_end_of_sentence(word, prev_word, next_word):
    features = extract_features(word, prev_word, next_word)
    return clf.predict(pd.DataFrame([features]))[0]

# Additional post-processing to filter out common abbreviations
def filter_abbreviations(predictions, words):
    filtered_predictions = []
    for i, (word, pred) in enumerate(zip(words, predictions)):
        if pred and word in ABBREVIATIONS:
            filtered_predictions.append(False)
        else:
            filtered_predictions.append(pred)
    return filtered_predictions

# Test with some examples
test_sentences = [
    "5.9 is not a good score. Dr. Smith. U.S.A. is a country."
]

for sentence in test_sentences:
    words = sentence.split()
    predictions = []
    for i, word in enumerate(words):
        prev_word = words[i-1] if i > 0 else ""
        next_word = words[i+1] if i < len(words) - 1 else ""
        is_end = predict_end_of_sentence(word, prev_word, next_word)
        predictions.append(is_end)

    filtered_predictions = filter_abbreviations(predictions, words)

    for i, (word, is_end) in enumerate(zip(words, filtered_predictions)):
        if is_end:
            print(f"Sentence: {sentence}")
            print(f"End detected at: {word}")
            print()
