# Credit-Card-Fraud-With-KNN

# 💳 Real-Time Fraud Detection Tool (Python + Tkinter)

An interactive **fraud detection simulation and analysis tool** built with **Python**, **Pandas**, **NumPy**, **scikit-learn**, and **Tkinter**.  
This project detects **potential fraudulent transactions** using a **feature-engineered nearest neighbor model**, visualized in a live ledger GUI.

---

## 🚀 Overview
This project simulates and analyzes financial transactions in real time, classifying each as **VALID** or **FRAUD** based on transaction behavior and balance discrepancies.  

It includes:
- A **custom fraud-detection algorithm** that models distance-based fraud likelihood using engineered financial features.
- **Feature engineering** that detects behavioral anomalies (e.g., account drainage, balance mismatches).
- **Class rebalancing** to counter dataset imbalance and improve fraud sensitivity.
- An **interactive GUI** for monitoring transactions and diagnostics in real time.

---

## 🧠 How It Works

### 🔹 1. Data Preprocessing
- Loads the **CCFraud Training Dataset** (CSV file).
- Engineers intelligent features:
  - `errorBalance` → captures mismatches between old balance, transaction amount, and new balance.
  - `drainedAccount` → identifies transactions where an account’s balance drops to zero.
- Handles **class imbalance** by oversampling fraud cases to achieve a more balanced dataset.

### 🔹 2. Model Logic
The algorithm predicts whether a transaction is **fraudulent** by:
1. Performing the same feature engineering on new input data.
2. Using a **k-nearest neighbors (distance-based)** comparison to the training set.  
3. Classifying the transaction as FRAUD if the average fraud likelihood among nearest neighbors exceeds 0.5.

### 🔹 3. Real-Time GUI
Built with **Tkinter**, the graphical interface:
- Displays transactions as they stream in.
- Highlights **VALID** (green) and **FRAUD** (red) predictions.
- Shows diagnostic info including nearest neighbor fraud ratios.
- Allows CSV import, progress tracking, and summary statistics of fraud detection.

---

## 🧩 Key Features

| Category | Description |
|-----------|-------------|
| **Feature Engineering** | Detects hidden fraud signals using engineered behavioral indicators |
| **Class Balancing** | Oversamples rare fraud cases to enhance model robustness |
| **Custom ML Logic** | Distance-based fraud detection using engineered numerical features |
| **Interactive GUI** | Live transaction feed with fraud highlighting and diagnostics |
| **CSV Upload Mode** | Batch-analyze real transaction data via file import |
| **Real-Time Ledger Simulation** | Simulates thousands of realistic transactions per session |

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.10+ |
| **Libraries** | `pandas`, `numpy`, `scikit-learn`, `tkinter`, `threading` |
| **ML Components** | MinMaxScaler, custom nearest-neighbor classifier |
| **Visualization** | Tkinter ScrolledText live ledger |
| **Data Source** | `CCFraud-TrainingData.csv` |

---

## 📊 Example Output (Live Ledger)

```

# Status     Type            Amount     Old Balance       New Balance       Diagnostics

VALID      PAYMENT      $   5,230.00$      8,500.00$      3,270.00 Neighbors: [0, 0, 0, 0, 0] (Avg: 0.00)
VALID      CASH_IN      $  12,450.00$     25,100.00$     37,550.00 Neighbors: [0, 0, 0, 0, 1] (Avg: 0.20)
FRAUD      TRANSFER     $ 210,000.00$    210,000.00$          0.00 Neighbors: [1, 1, 1, 1, 1] (Avg: 1.00)

````

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-tool.git
cd fraud-detection-tool

# Install dependencies
pip install pandas numpy scikit-learn
````

Make sure to update the path in the script to point to your **training dataset**:

```python
training_file_path = "path/to/CCFraud-TrainingData.csv"
```

---

## ▶️ Run the Application

```bash
python fraud_detection_gui.py
```

Once launched:

1. Click **“Load Transaction CSV”** to select your data file.
2. Click **“Start Detection”** to begin processing transactions.
3. Watch the live ledger populate with **VALID/FRAUD** transactions and diagnostics.

---

## 🧮 Model Summary

| Metric                  | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **Algorithm**           | Distance-based nearest-neighbor fraud probability            |
| **Feature Engineering** | `errorBalance`, `drainedAccount`, one-hot encoding of `type` |
| **Scaler**              | `MinMaxScaler` for normalized comparison                     |
| **Window Size (k)**     | 5 nearest neighbors                                          |
| **Output**              | Binary classification: VALID / FRAUD                         |

---

## 💡 Future Improvements

* Integrate a **trained classifier (Random Forest or XGBoost)** for improved accuracy
* Add **ROC/AUC evaluation** and **confusion matrix metrics**
* Incorporate **real-time data ingestion** from APIs or message queues
* Implement a **database backend** for logging results

---

## 📜 License

Released under the **MIT License** — open for learning, modification, and educational research.

---

### 👤 Author

**Nico Moran**
📈 Quantitative Finance & Machine Learning Enthusiast
📧 [nxcomoran@gmail.com](mailto:nxcomoran@gmail.com)
---

```
