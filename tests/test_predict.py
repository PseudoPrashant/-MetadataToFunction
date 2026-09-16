import sys
import os
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import format_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "function_name_model.pkl")

def test_prediction_works_if_model_exists():
    if not os.path.exists(MODEL_PATH):
        # Skip test if model isn't trained yet
        return
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    formatted = format_metadata(
        "Adds two integers", "int a int b", "int", "MathUtils", "add sum", 2
    )
    
    prediction = model.predict([formatted])
    
    # We expect 'addNumbers' based on the original README examples
    assert prediction[0] == "addNumbers"
