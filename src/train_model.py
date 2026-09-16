import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from utils import format_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "trainingDataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "function_name_model.pkl")
ONNX_PATH = os.path.join(BASE_DIR, "models", "function_name_model.onnx")

df = pd.read_csv(DATA_PATH)

df["combined"] = df.apply(lambda row: format_metadata(
    row["description"], row["parameters"], row["return_type"],
    row["library"], row["keywords"], row["param_count"]
), axis=1)

X = df["combined"]
y = df["function_name"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf",   MultinomialNB())
])

param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [0.1, 0.5, 1.0]
}

print("Starting GridSearchCV to find optimal hyperparameters...")
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Best parameters found: {grid.best_params_}")

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy with best model: {acc * 100:.2f}%")

# Save as Pickle
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
size = os.path.getsize(MODEL_PATH)
print(f"Pickle Model saved successfully! Size: {size / 1024:.1f} KB")

# Export as ONNX
initial_type = [('input_text', StringTensorType([None, 1]))]
onx = convert_sklearn(best_model, initial_types=initial_type)
with open(ONNX_PATH, "wb") as f:
    f.write(onx.SerializeToString())
onnx_size = os.path.getsize(ONNX_PATH)
print(f"ONNX Model saved successfully! Size: {onnx_size / 1024:.1f} KB")
