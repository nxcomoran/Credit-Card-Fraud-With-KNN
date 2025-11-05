import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import scrolledtext
import random
import time
import threading

# --- 1. MODEL SETUP (Done once at the start) ---
print("--- Initializing Model ---")

# --- Configuration ---
training_file_path = 'D:\\Python\\CCFraud-TrainingData.csv'
# <<< CRITICAL FIX 1: Add our new engineered features to the list
feature_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "errorBalance", "drainedAccount"]
target_column = "isFraud"

# --- Load Data ---
try:
    df_train = pd.read_csv(training_file_path)
    print(f"Training dataset loaded successfully from '{training_file_path}'")
except FileNotFoundError:
    print(f"Error: Training dataset file not found. Update 'training_file_path'.")
    exit()

# --- CRITICAL FIX 2: FEATURE ENGINEERING ---
# Create new 'smart' features that explicitly describe fraudulent behavior.
print("\nPerforming Feature Engineering...")
# This feature highlights discrepancies in the balance updates.
df_train['errorBalance'] = df_train.newbalanceOrig + df_train.amount - df_train.oldbalanceOrg
# This feature is a '1' if an account was wiped out, '0' otherwise.
df_train['drainedAccount'] = ((df_train.oldbalanceOrg > 0) & (df_train.newbalanceOrig == 0)).astype(int)

# --- Address Class Imbalance using the feature-engineered dataframe ---
print("Addressing class imbalance...")
fraud_cases = df_train[df_train['isFraud'] == 1]
non_fraud_cases = df_train[df_train['isFraud'] == 0]
df_fraud_oversampled = pd.concat([fraud_cases] * 70, ignore_index=True)
df_train_balanced = pd.concat([non_fraud_cases, df_fraud_oversampled], ignore_index=True)
df_train_balanced = df_train_balanced.sample(frac=1).reset_index(drop=True)
print("Balanced dataset created.")

# --- Use the NEW BALANCED and ENGINEERED data for training ---
X_train = df_train_balanced[feature_columns].copy()
y_train_np = df_train_balanced[target_column].values.flatten()

# One-Hot Encode and get feature names
X_train_processed = pd.get_dummies(X_train, columns=['type'], prefix='type', drop_first=True)
processed_feature_names = X_train_processed.columns.tolist()
print(f"Final Model features: {processed_feature_names}")

# --- Fit the Scaler on the BALANCED and ENGINEERED data ---
scaler = MinMaxScaler()
X_train_scaled_np = scaler.fit_transform(X_train_processed)
print("Model setup and data scaling complete.")


# --- 2. CORE PREDICTION LOGIC ---
def predict_and_diagnose_risk(transaction_data, model_scaler, model_features, training_X, training_y, k=5):
    new_df = pd.DataFrame([transaction_data])

    # <<< CRITICAL FIX 3: Replicate the EXACT SAME feature engineering on new data
    new_df['errorBalance'] = new_df.newbalanceOrig + new_df.amount - new_df.oldbalanceOrg
    new_df['drainedAccount'] = ((new_df.oldbalanceOrg > 0) & (new_df.newbalanceOrig == 0)).astype(int)
    
    new_df_processed = pd.get_dummies(new_df, columns=['type'], prefix='type', drop_first=True)
    new_df_aligned = new_df_processed.reindex(columns=model_features, fill_value=0)
    new_df_scaled = model_scaler.transform(new_df_aligned)

    squared_diffs = (training_X - new_df_scaled)**2
    distances = (squared_diffs.sum(axis=1))**0.5
    neighbor_indices = np.argpartition(distances, k)[:k]
    
    neighbor_labels = training_y[neighbor_indices]
    avg_fraud_chance = np.mean(neighbor_labels)
    
    if avg_fraud_chance > 0.5:
        prediction = "FRAUD" 
    else:
        prediction = "VALID"
    
    return {"prediction": prediction, "avg_fraud_chance": avg_fraud_chance, "neighbor_labels": neighbor_labels.tolist()}

# --- 3. TRANSACTION GENERATORS (No changes needed) ---
def generate_random_transaction():
    trans_type = random.choice(["CASH_IN", "CASH_OUT", "DEBIT", "TRANSFER", "PAYMENT"])
    old_balance = round(random.uniform(1.0, 1000000.0), 2)
    amount = round(random.uniform(1.0, old_balance), 2)
    new_balance = old_balance + amount if trans_type == "CASH_IN" else old_balance - amount
    return {"step": random.randint(1, 999), "type": trans_type, "amount": amount, "oldbalanceOrg": old_balance, "newbalanceOrig": new_balance}

def generate_suspect_transaction():
    print("Injecting a high-risk transaction...")
    trans_type = random.choice(["CASH_OUT", "TRANSFER"])
    balance = round(random.uniform(1000.0, 500000.0), 2)
    return {"step": random.randint(1, 999), "type": trans_type, "amount": balance, "oldbalanceOrg": balance, "newbalanceOrig": 0.0}

# --- 4. GUI APPLICATION (No changes needed) ---
class FraudLedgerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Fraud Detection Ledger")
        self.root.geometry("1000x700")

        self.ledger = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10), bg="black")
        self.ledger.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.ledger.tag_config('VALID', foreground='spring green', font=("Consolas", 10, "bold"))
        self.ledger.tag_config('FRAUD', foreground='red', font=("Consolas", 10, "bold"))
        self.ledger.tag_config('DIAG', foreground='cyan')
        self.ledger.tag_config('HEADER', foreground='white', font=("Consolas", 10, "bold"))

        header = f"{'Status':<10}{'Type':<12}{'Amount':>15}{'Old Balance':>18}{'New Balance':>18} {'Diagnostics'}\n"
        self.ledger.insert(tk.END, header, 'HEADER')
        self.ledger.insert(tk.END, "="*110 + "\n", 'HEADER')

        self.transaction_counter = 0
        self.injection_threshold = random.randint(0, 80)
        self.running = True
        self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def add_transaction_to_ledger(self, transaction, diagnostics):
        status = diagnostics["prediction"]
        trans_str = (f"{status:<10}{transaction['type']:<12}${transaction['amount']:>13,.2f}"
                     f"${transaction['oldbalanceOrg']:>16,.2f}${transaction['newbalanceOrig']:>16,.2f} ")
        self.ledger.insert(tk.END, trans_str, status)
        diag_info = f"Neighbors: {diagnostics['neighbor_labels']} (Avg: {diagnostics['avg_fraud_chance']:.2f})\n"
        self.ledger.insert(tk.END, diag_info, 'DIAG')
        self.ledger.see(tk.END)

    def run_simulation(self):
        while self.running:
            self.transaction_counter += 1
            if self.transaction_counter >= self.injection_threshold:
                new_trans = generate_suspect_transaction()
                self.transaction_counter = 0
                self.injection_threshold = random.randint(0, 80)
            else:
                new_trans = generate_random_transaction()
            diagnostics = predict_and_diagnose_risk(new_trans, scaler, processed_feature_names, X_train_scaled_np, y_train_np)
            self.root.after(0, self.add_transaction_to_ledger, new_trans, diagnostics)
            time.sleep(random.uniform(0.1, 0.5))

    def on_closing(self):
        print("Closing application...")
        self.running = False
        self.root.destroy()

# --- 5. LAUNCH THE APPLICATION ---
if __name__ == "__main__":
    print("\nModel ready. Launching GUI...")
    root = tk.Tk()
    app = FraudLedgerApp(root)
    root.mainloop()