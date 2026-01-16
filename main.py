import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importing the cleaning function from our local preprocessing file
from preprocessing import clean_text

def run_project():
    # --- Step 1: Setup Output Directory ---
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")

    # --- Step 2: Load IMDb Dataset ---
    print("\n--- Step 2: Loading IMDb data ---")
    
    local_filename = 'IMDB Dataset.csv'
    # Updated backup URL
    data_url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    
    try:
        # Check if the file exists locally first
        if os.path.exists(local_filename):
            print(f"Loading data from local file: {local_filename}")
            df = pd.read_csv(local_filename)
        else:
            print(f"Local file not found. Attempting to download from: {data_url}")
            df = pd.read_csv(data_url)
        
        # Limiting to 5000 rows for speed. 
        # DELETE the line below if you want to use all 50,000 rows (will be much slower)
        df = df.head(5000) 
        
        print(f"Successfully loaded {len(df)} reviews.")
        
        # Mapping 'positive' to 1 and 'negative' to 0
        df['sentiment'] = df['sentiment'].str.lower().map({'positive': 1, 'negative': 0})
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nFATAL ERROR: Please make sure 'IMDB Dataset.csv' is in the project folder.")
        return

    # --- Step 3: Preprocessing ---
    print("--- Step 3: Cleaning text data (this may take a moment) ---")
    df['review_cleaned'] = df['review'].apply(clean_text)

    # --- Step 4: Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_cleaned'], df['sentiment'], test_size=0.2, random_state=42
    )

    # --- Step 5: Feature Extraction (TF-IDF) ---
    vectorizer = TfidfVectorizer(max_features=5000) # Limiting features for efficiency
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # --- Step 6: Model Training ---
    # SVM
    print("--- Step 4: Training SVM Model (Linear Kernel) ---")
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)

    # Naive Bayes
    print("--- Step 5: Training Naive Bayes Model ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    # --- Step 7: Evaluation & Visualization ---
    print("\n--- FINAL RESULTS (REAL DATA) ---")
    acc_svm = accuracy_score(y_test, y_pred_svm)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    
    print(f"SVM Accuracy: {acc_svm:.4f}")
    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")

    # Save detailed report to text file
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write("SVM RESULTS (IMDb Data):\n")
        f.write(classification_report(y_test, y_pred_svm))
        f.write("\n\nNAIVE BAYES RESULTS (IMDb Data):\n")
        f.write(classification_report(y_test, y_pred_nb))

    # --- GENERATING SEPARATE PLOTS ---

    # Plot 1: Accuracy Comparison (The "two high colored blocks" chart)
    plt.figure(figsize=(8, 6))
    model_names = ['SVM', 'Naive Bayes']
    accuracy_values = [acc_svm, acc_nb]
    sns.barplot(x=model_names, y=accuracy_values, palette='magma')
    plt.ylim(0, 1.1)
    plt.title('Accuracy Comparison: SVM vs Naive Bayes')
    plt.ylabel('Accuracy Score')
    # Add values on top of bars
    for i, val in enumerate(accuracy_values):
        plt.text(i, val + 0.02, f'{val:.4f}', ha='center', fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    print("Saved: accuracy_comparison.png")

    # Plot 2: SVM Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_svm.png'))
    plt.close()
    print("Saved: confusion_matrix_svm.png")

    # Plot 3: Naive Bayes Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, cmap='Greens', fmt='d')
    plt.title('Confusion Matrix - Naive Bayes')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_nb.png'))
    plt.close()
    print("Saved: confusion_matrix_nb.png")

    print(f"\nSuccess! All individual plots and reports saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    run_project()