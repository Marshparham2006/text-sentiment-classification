
Text Sentiment Classification using Machine Learning
----------------------------------------------------

This script demonstrates a supervised machine learning workflow for binary sentiment classification 
of short text reviews. The dataset is defined directly in the code. The workflow includes
preprocessing, feature extraction, model training, evaluation, and prediction on new samples.
"""

# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Define Dataset
# -------------------------------
# Small manually created dataset of text reviews with sentiment labels
reviews = [
    "I love this product",
    "This is terrible",
    "Amazing experience",
    "I hate it",
    "Very good quality",
    "Not good at all",
    "Excellent service",
    "Worst experience ever"
]

sentiments = [
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Negative",
    "Positive",
    "Negative"
]

# -------------------------------
#  Text Vectorization
# -------------------------------
# Convert text into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(reviews)
y = sentiments

# -------------------------------
#  Train-Test Split
# -------------------------------
# Split dataset into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.25, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
#  Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")

# -------------------------------
#  Predict New Reviews
# -------------------------------
new_reviews = [
    "I really love this",
    "This product is awful",
    "Fantastic quality and service",
    "I am not satisfied"
]

# Transform new reviews using the same vectorizer
new_vect = vectorizer.transform(new_reviews)
predictions = model.predict(new_vect)

# Display prediction results
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: '{review}' -> Sentiment: {sentiment}")
