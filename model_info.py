import pickle
import os
import time

# Load model
with open("function_name_model.pkl", "rb") as f:
    model = pickle.load(f)

# ── CHECK 1: Model Size ───────────────────────────────
size_bytes = os.path.getsize("function_name_model.pkl")
size_kb    = size_bytes / 1024
size_mb    = size_kb / 1024

print("===== Model Size =====")
print(f"Size: {size_bytes} bytes")
print(f"Size: {size_kb:.1f} KB")
print(f"Size: {size_mb:.3f} MB")

# ── CHECK 2: Inference Speed ──────────────────────────
test_input = ["Adds two integers int a int b int MathUtils add sum 2"]

print("\n===== Inference Speed =====")
start = time.time()
for i in range(1000):
    model.predict(test_input)
end = time.time()

total_ms  = (end - start) * 1000
single_ms = total_ms / 1000
print(f"1000 predictions took : {total_ms:.1f} ms")
print(f"Single prediction took: {single_ms:.4f} ms")

# ── CHECK 3: Deployment Suitability ──────────────────
print("\n===== Deployment Suitability =====")
print(f"Model size under 1MB  : {size_mb < 1}")
print(f"Single prediction < 1ms: {single_ms < 1}")

if size_mb < 1 and single_ms < 1:
    print("\nModel is lightweight and suitable for Android deployment!")
else:
    print("\nModel may need optimization")


