import pickle
import os
import sys
from utils import format_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "function_name_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run 'python src/train_model.py' first to generate the model.")
    sys.exit(1)

print("Model loaded successfully!")

print("\n--- Try your own input ---")
while True:
    print("\nEnter metadata fields (or type 'quit' at any prompt to exit):")
    description = input("Description (e.g. Adds two integers): ")
    if description.lower() == "quit": break
    
    parameters = input("Parameters (e.g. int a int b): ")
    if parameters.lower() == "quit": break
    
    return_type = input("Return Type (e.g. int): ")
    if return_type.lower() == "quit": break
    
    library = input("Library (e.g. MathUtils): ")
    if library.lower() == "quit": break
    
    keywords = input("Keywords (e.g. add sum): ")
    if keywords.lower() == "quit": break
    
    param_count = input("Param Count (e.g. 2): ")
    if param_count.lower() == "quit": break

    formatted_input = format_metadata(description, parameters, return_type, library, keywords, param_count)
    prediction = model.predict([formatted_input])
    print(f"\n=> Predicted function name: {prediction[0]}")
