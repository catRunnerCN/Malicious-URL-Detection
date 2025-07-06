# Malicious URL Detection Using LSTM and Handcrafted Features

This project builds a hybrid deep learning model that classifies URLs as **malicious or benign**. It combines LSTM-based sequence learning from raw URLs with handcrafted statistical and structural features.

## Features

* LSTM-based neural network on tokenized URL sequences
* Extracted handcrafted features (entropy, length, IP presence, special chars, etc.)
* Label encoding for binary classification (good/bad)
* Fusion of sequence and structured features
* Evaluation via confusion matrix, classification report, and ROC-AUC

## Dataset

The input CSV file should contain at least:

* `URL`: the full URL string
* `Label`: either `good` or `bad`

> Example path used in code:
> `H:/s_and_r/.venv/ai_training_data(2).csv`

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn
* tqdm

## How to Run

1. Clone this repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your dataset CSV with `URL` and `Label` columns.
4. Run the main script:

   ```bash
   python malicious_url_detector.py
   ```

> The script will:
>
> * Extract features from URLs
> * Train an LSTM + handcrafted-feature fusion model
> * Output performance metrics and plots
> * Export extracted features to CSV

## Feature Extraction

Custom URL features include:

* Entropy of URL
* URL length, depth, number of digits
* Presence of `http`, `https`, or IP address
* Count of special characters: `?`, `@`, `&`, etc.
* Domain and subdomain length
* Fragment and parameter counts

## 📊 Output

* Confusion Matrix
* Classification Report (precision, recall, F1-score)
* ROC Curve with AUC score
* CSV: `url_extracted_features.csv`, `url_extracted_features_with_label.csv`
