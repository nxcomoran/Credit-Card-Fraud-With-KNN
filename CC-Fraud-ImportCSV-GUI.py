import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk

import threading
import time

# --- 1. MODEL SETUP (Done once at the start) ---
print("--- Initializing Model ---")

# --- Configuration ---
# training_file_path = 'D:\\Python\\CCFraud-TrainingData.csv' CHANGE TO LOCAL FILE PATH
feature_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "errorBalance", "drainedAccount"]
target_column = "isFraud"

# --- Load Data ---
try:
    df_train = pd.read_csv(training_file_path)
    print(f"Training dataset loaded successfully from '{training_file_path}'")
except FileNotFoundError:
    print(f"Error: Training dataset file not found. Update 'training_file_path'.")
    exit()

# --- Feature Engineering on Training Data ---
print("\nPerforming Feature Engineering...")
df_train['errorBalance'] = df_train.newbalanceOrig + df_train.amount - df_train.oldbalanceOrg
df_train['drainedAccount'] = ((df_train.oldbalanceOrg > 0) & (df_train.newbalanceOrig == 0)).astype(int)

# --- Address Class Imbalance ---
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

# --- Fit the Scaler ---
scaler = MinMaxScaler()
X_train_scaled_np = scaler.fit_transform(X_train_processed)
print("Model setup and data scaling complete.")


# --- 2. CORE PREDICTION LOGIC ---
def predict_and_diagnose_risk(transaction_data, model_scaler, model_features, training_X, training_y, k=5):
    new_df = pd.DataFrame([transaction_data])

    # Replicate the EXACT SAME feature engineering on the new data
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
    prediction = "FRAUD" if avg_fraud_chance > 0.5 else "VALID"
    
    return {"prediction": prediction, "avg_fraud_chance": avg_fraud_chance, "neighbor_labels": neighbor_labels.tolist()}

# --- 3. GUI APPLICATION ---
class FraudGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Detection Tool")
        self.root.geometry("1000x700")

        # --- Top frame for controls ---
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.load_button = tk.Button(control_frame, text="Load Transaction CSV", command=self.load_csv_file)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.start_button = tk.Button(control_frame, text="Start Detection", command=self.start_processing_thread, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(control_frame, text="No file loaded.")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # --- NEW: Label for the final summary ---
        self.summary_label = tk.Label(control_frame, text="", font=("Segoe UI", 9, "bold"))
        self.summary_label.pack(side=tk.LEFT, padx=20)

        # --- Progress Bar ---
        self.progress_bar = ttk.Progressbar(root, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # --- Scrolled text for ledger ---
        self.ledger = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10), bg="black")
        self.ledger.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.setup_ledger()

        # --- App state variables ---
        self.loaded_df = None
        self.fraud_count = 0 # NEW: Counter for fraud cases
        self.running_thread = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ledger(self):
        """Clears the ledger and sets up tags and header."""
        self.ledger.configure(state='normal')
        self.ledger.delete('1.0', tk.END)
        self.ledger.tag_config('VALID', foreground='spring green', font=("Consolas", 10, "bold"))
        self.ledger.tag_config('FRAUD', foreground='red', font=("Consolas", 10, "bold"))
        self.ledger.tag_config('DIAG', foreground='cyan')
        self.ledger.tag_config('HEADER', foreground='white', font=("Consolas", 10, "bold"))
        header = f"{'Status':<10}{'Type':<12}{'Amount':>15}{'Old Balance':>18}{'New Balance':>18} {'Diagnostics'}\n"
        self.ledger.insert(tk.END, header, 'HEADER')
        self.ledger.insert(tk.END, "="*110 + "\n", 'HEADER')
        self.ledger.configure(state='disabled')

    def load_csv_file(self):
        """Opens a file dialog to load a CSV and prepares the app for processing."""
        file_path = filedialog.askopenfilename(
            title="Select a transaction file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not file_path:
            return

        try:
            self.loaded_df = pd.read_csv(file_path)
            required_cols = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
            if not all(col in self.loaded_df.columns for col in required_cols):
                raise ValueError(f"CSV must contain the columns: {required_cols}")

            self.file_label.config(text=file_path.split('/')[-1])
            self.start_button.config(state=tk.NORMAL)
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = len(self.loaded_df)
            self.summary_label.config(text="") # Clear previous results
            self.setup_ledger()
            messagebox.showinfo("Success", f"Successfully loaded {len(self.loaded_df)} transactions.")
        except Exception as e:
            messagebox.showerror("Error Loading File", f"An error occurred: {e}")
            self.loaded_df = None

    def start_processing_thread(self):
        """Starts the file processing in a separate thread."""
        if self.loaded_df is None:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return

        # NEW: Reset fraud counter before starting
        self.fraud_count = 0

        self.start_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)
        
        self.running_thread = threading.Thread(target=self.process_file, daemon=True)
        self.running_thread.start()

    def process_file(self):
        """Iterates through the DataFrame, runs predictions, and counts fraud."""
        for index, row in self.loaded_df.iterrows():
            transaction_dict = row.to_dict()
            diagnostics = predict_and_diagnose_risk(
                transaction_dict, scaler, processed_feature_names, X_train_scaled_np, y_train_np
            )

            # NEW: Increment the counter if fraud is detected
            if diagnostics["prediction"] == "FRAUD":
                self.fraud_count += 1
            
            self.root.after(0, self.update_gui, transaction_dict, diagnostics, index + 1)
           # time.sleep(0.0) 

        self.root.after(0, self.processing_complete)

    def update_gui(self, transaction, diagnostics, progress_value):
        """Updates the ledger and progress bar."""
        self.ledger.configure(state='normal')
        status = diagnostics["prediction"]
        trans_str = (f"{status:<10}{transaction['type']:<12}${transaction['amount']:>13,.2f}"
                     f"${transaction['oldbalanceOrg']:>16,.2f}${transaction['newbalanceOrig']:>16,.2f} ")
        self.ledger.insert(tk.END, trans_str, status)
        diag_info = f"Neighbors: {diagnostics['neighbor_labels']} (Avg: {diagnostics['avg_fraud_chance']:.2f})\n"
        self.ledger.insert(tk.END, diag_info, 'DIAG')
        self.ledger.see(tk.END)
        self.ledger.configure(state='disabled')
        self.progress_bar['value'] = progress_value

    def processing_complete(self):
        """Called when processing finishes to re-enable controls and show summary."""
        self.load_button.config(state=tk.NORMAL)

        # --- NEW: Calculate and display the final summary ---
        total_transactions = len(self.loaded_df)
        if total_transactions > 0:
            percentage = (self.fraud_count / total_transactions) * 100
            summary_text = f"Result: {self.fraud_count} of {total_transactions} transactions flagged ({percentage:.2f}% fraud)."
            self.summary_label.config(text=summary_text)
            if self.fraud_count > 0:
                self.summary_label.config(fg="red")
            else:
                self.summary_label.config(fg="green")

        messagebox.showinfo("Complete", "Fraud detection process has finished.")

    def on_closing(self):
        """Handles closing the application."""
        if messagebox.askokcancel("Quit", "Do you want to exit?"):
            self.root.destroy()

# --- 4. LAUNCH THE APPLICATION ---
if __name__ == "__main__":
    print("\nModel ready. Launching GUI...")
    root = tk.Tk()
    app = FraudGuiApp(root)
    root.mainloop()