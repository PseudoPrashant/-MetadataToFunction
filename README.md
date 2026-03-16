# MetadataToFunction

> A lightweight machine learning model that predicts Python function names from metadata — designed for Android and low-computation deployment.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Deployment](#deployment)

---

## Overview

MetadataToFunction solves a simple but useful problem — given metadata about a function (its description, parameters, return type, library, and keywords), predict what the function should be named.

**Example:**

| Input Metadata | Predicted Output |
|---|---|
| "Adds two integers int a int b int MathUtils add sum 2" | `addNumbers` |
| "Reads file from disk str filepath str FileUtils read file 1" | `readFile` |
| "Converts string to uppercase str text str StringUtils uppercase 1" | `toUpperCase` |

This project was built as part of an assignment to design a lightweight ML model suitable for deployment on Android or low-computation devices.

---

## How It Works

All metadata columns are combined into a single text string, which is then passed through a two-step pipeline:

```
Metadata Input (combined string)
        │
        ▼
┌───────────────────┐
│  TF-IDF Vectorizer │  converts text → numbers
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Multinomial NB   │  predicts function name
└───────────────────┘
        │
        ▼
  Predicted Function Name
```

**TF-IDF (Term Frequency - Inverse Document Frequency)** scores each word by how important it is — rare and specific words like "celsius" or "encrypt" score high, while common words like "function" or "returns" score low.

**Multinomial Naive Bayes** learns the probability of each function name given each word during training. At prediction time it multiplies those probabilities and picks the function name with the highest score.

---

## Dataset

- **File:** `realistic_function_dataset_2160_rows.csv`
- **Rows:** 1080 unique examples
- **Columns:** 7
- **Unique function names:** 90
- **Library categories:** 25

| Column | Description | Example |
|---|---|---|
| `description` | What the function does | "Adds two integers" |
| `parameters` | Input parameters | "int a, int b" |
| `return_type` | Return type | "int" |
| `library` | Library or module name | "MathUtils" |
| `keywords` | Relevant keywords | "add, sum, arithmetic" |
| `param_count` | Number of parameters | 2 |
| `function_name` | Target label | "addNumbers" |

**Library categories include:**
MathUtils, StringUtils, FileUtils, NetworkService, DateUtils, ArrayUtils, CryptoUtils, DatabaseService, AuthService, JsonUtils, Validator, Logger, CacheService, ImageProcessor, AudioProcessor, and 10 more.

---

## Model

| Property | Value |
|---|---|
| Algorithm | TF-IDF + Multinomial Naive Bayes |
| Test Accuracy | **98.61%** |
| Model Size | **391.2 KB** |
| Inference Speed | **0.45ms per prediction** |
| Saved Format | `.pkl` (Python pickle) |

---

## Project Structure

```
MetadataToFunction/
    ├── realistic_function_dataset_2160_rows.csv  # training dataset
    ├── train_model.py                            # trains and saves the model
    ├── predict.py                                # loads model and runs predictions
    ├── model_info.py                             # checks model size and speed
    ├── README.md                                 # this file
    └── .gitignore                                # excludes cache and model file
```

> Note: `function_name_model.pkl` is not included in the repo. Generate it by running `train_model.py`.

---

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/PseudoPrashant/MetadataToFunction.git
cd MetadataToFunction
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn
```

**3. Train the model**
```bash
python train_model.py
```

This will generate `function_name_model.pkl` in your folder.

---

## Usage

### Running predictions

```bash
python predict.py
```

**Example output:**
```
Model loaded successfully!
Predicted function name: addNumbers

--- Try your own input ---
Enter metadata (or 'quit' to exit): Reads file from disk str filepath str FileUtils read file 1
Predicted function name: readFile
```

### Input format

Combine your metadata into one string in this order:
```
"{description} {parameters} {return_type} {library} {keywords} {param_count}"
```

**Example:**
```python
input = "Sends HTTP GET request str url HttpResponse NetworkService http request 1"
# → predicts: sendGetRequest
```

### Checking model size and speed

```bash
python model_info.py
```

**Output:**
```
===== Model Size =====
Size: 391.2 KB
Size: 0.382 MB

===== Inference Speed =====
1000 predictions took : 454.4 ms
Single prediction took: 0.4544 ms

===== Deployment Suitability =====
Model size under 1MB   : True
Single prediction < 1ms: True

✅ Model is lightweight and suitable for Android deployment!
```

---

## Results

| Metric | Result |
|---|---|
| Training rows | 864 |
| Testing rows | 216 |
| Test accuracy | **98.61%** |
| Classes with f1 ≥ 90% | 82 out of 90 |
| Classes with f1 < 80% | 1 (capitalizeWords) |

---

## Deployment

This model is designed to run on Android and low-computation devices because:

- **Size:** 391 KB — small enough to bundle inside any mobile app
- **Speed:** 0.45ms per prediction — feels instant even on low-end devices
- **No internet required:** runs fully offline once the model is loaded
- **No GPU needed:** Naive Bayes runs on CPU only

To deploy on Android, export the model using ONNX or use a Python-to-Android bridge like Chaquopy.

---

## Dependencies

```
pandas
scikit-learn
```

---

*Built as part of a Lightweight ML Assignment — Function Name Prediction from Metadata*
