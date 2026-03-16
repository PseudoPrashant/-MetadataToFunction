import pickle

with open("function_name_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")

print("\n--- Try your own input ---")
while True:
    user_input = input("Enter metadata (or 'quit' to exit): ")
    if user_input.lower() == "quit":
        break
    prediction = model.predict([user_input])
    print(f"Predicted function name: {prediction[0]}\n")


