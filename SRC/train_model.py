import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the prepared HC3 dataset
df = pd.read_csv("data/processed/hc3_human_ai.csv")

# Remove missing rows just in case
df = df.dropna(subset=["text", "label"])

# Keep only the columns we need
X = df["text"]
y = df["label"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build a simple text classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", SGDClassifier(loss="log_loss", random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/human_ai_model.joblib")
print("\nModel saved to models/human_ai_model.joblib")