import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from utils import format_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "trainingDataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "function_name_model.pkl")

df = pd.read_csv(DATA_PATH) #dataFrame variable 

df["combined"] = df.apply(lambda row: format_metadata(
    row["description"], row["parameters"], row["return_type"],
    row["library"], row["keywords"], row["param_count"]
), axis=1)

#print(df["combined"].iloc[1])

# print(df.head(1))   

X = df["combined"]
y = df["function_name"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print(f"Training rows : {len(X_train)}")
# print(f"Testing rows  : {len(X_test)}")

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf",   MultinomialNB())
])
model.fit(X_train, y_train)
print("Model trained successfully!")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

size = os.path.getsize(MODEL_PATH)
print(f"Model saved successfully!")
print(f"Model size: {size / 1024:.1f} KB")

