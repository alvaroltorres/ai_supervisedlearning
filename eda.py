# Step 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # <-- Import the os module

# --- Add this after importing libraries ---
# Define the folder name for saving plots
eda_folder = 'eda'
# Create the folder if it doesn't exist
try:
    os.makedirs(eda_folder, exist_ok=True)
    print(f"Plots will be saved in the '{eda_folder}' directory.")
except OSError as error:
    print(f"Error creating directory {eda_folder}: {error}")
    # Decide how to handle the error, e.g., exit or proceed without saving
    eda_folder = None # Set folder to None to skip saving if creation fails
    print("Proceeding without saving plots due to directory creation error.")
# ---

# Set plotting style
sns.set(style="whitegrid")

print("Libraries imported successfully.")

# Step 1: Ler dados (Read Data)
# --- IMPORTANT: Adjust the path to where your 'train.csv' file is located ---
try:
    # Assuming the script is run from a directory containing the dataset file
    # or provide the full path if it's elsewhere.
    file_path = 'train_test_dataset/train.csv'
    if not os.path.exists(file_path):
         # Fallback if not in current dir, try a common relative path like 'data/'
         alt_path = os.path.join('data', 'train.csv')
         if os.path.exists(alt_path):
             file_path = alt_path
         else:
             # Try one level up if maybe running from a 'scripts' folder
             alt_path = os.path.join('..', 'data', 'train.csv')
             if os.path.exists(alt_path):
                 file_path = alt_path
             else:
                # Default back if others fail, let the original error happen
                file_path = 'train.csv'


    df_train = pd.read_csv(file_path)
    print(f"\nSuccessfully loaded data from: {file_path}")
except FileNotFoundError:
    print(f"\nError: Could not find the file at '{file_path}' or common alternatives.")
    print("Please ensure 'train.csv' is in the correct location or update the 'file_path' variable.")
    exit() # Exit if file not found

print("\n--- 1. Data Loading ---")
print("Shape of the training data (rows, columns):", df_train.shape)
print("\nFirst 5 rows of the data:")
print(df_train.head())

# Identify features and targets based on typical Kaggle structure for this dataset
# Usually, the last 7 columns are the target labels. Let's verify column names.
print("\nColumn Names:")
print(df_train.columns)

# Assuming standard names for the target columns based on UCI/Kaggle
target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
# Check if all expected target columns exist
actual_target_columns = [col for col in target_columns if col in df_train.columns]
if len(actual_target_columns) != len(target_columns):
    print("\nWarning: Not all expected target columns found. Using available ones.")
    target_columns = actual_target_columns

# Identify feature columns (all columns except targets and potentially an 'id' column if present)
potential_id_col = 'id' # Common name for ID columns in Kaggle datasets
cols_to_exclude = target_columns + ([potential_id_col] if potential_id_col in df_train.columns else [])
feature_columns = [col for col in df_train.columns if col not in cols_to_exclude]


print(f"\nIdentified {len(feature_columns)} feature columns.")
print(f"Identified {len(target_columns)} target columns (defect types): {', '.join(target_columns)}")


# Step 2: Explorar Variaveis (Explore Variables) - EDA

print("\n--- 2. Exploratory Data Analysis (EDA) ---")

# 2.a: Data Types and Non-Null Counts
print("\n--- 2.a Data Types and Non-Null Values ---")
print(df_train.info())
# Observation: This will show Dtype for each column and non-null counts.
# Pay attention to whether counts match the total number of rows.

# 2.b: Missing Values Check (`2.1 Valores em falta`)
print("\n--- 2.b Missing Values per Column ---")
missing_values = df_train.isnull().sum()
missing_values_filtered = missing_values[missing_values > 0] # Only show columns with missing values
if not missing_values_filtered.empty:
    print(missing_values_filtered)
    print(f"Total missing values found: {missing_values.sum()}")
else:
    print("No missing values found in the training data.")
# Observation: Note if any columns have missing data. The original UCI dataset description
# often states no missing values, but it's crucial to verify.

# 2.c: Basic Statistical Summary (`2.1 Valores errados`, `ver max, min, media, etc`)
print("\n--- 2.c Statistical Summary for Numerical Features ---")
# Use .describe() on feature columns only
stats_summary = df_train[feature_columns].describe().T # Transpose (.T) for better readability
print(stats_summary)
# Save summary statistics to a CSV file in the EDA folder
if eda_folder:
    try:
        stats_filename = os.path.join(eda_folder, 'numerical_features_summary.csv')
        stats_summary.to_csv(stats_filename)
        print(f"\nSaved statistical summary to: {stats_filename}")
    except Exception as e:
        print(f"\nCould not save statistical summary: {e}")
# Observation: Look at min/max for unreasonable values. Check the mean vs median (50%)
# to get an idea of skewness. Note the range (max-min) and standard deviation (std)
# for different features to assess scale differences ('2.2 Normalizar?').

# 2.d: Target Variable Analysis (Class Distribution) (`2.3 ... com a target var`)
print("\n--- 2.d Target Variable Distribution (Defect Types) ---")
target_counts = df_train[target_columns].sum().sort_values(ascending=False)
print("Counts for each defect type:")
print(target_counts)

# Calculate the number of samples with NO defects
no_defect_count = df_train[df_train[target_columns].sum(axis=1) == 0].shape[0]
print(f"\nNumber of samples with NO defects: {no_defect_count}")
print(f"Total number of samples: {df_train.shape[0]}")
print(f"Number of samples with at least one defect: {df_train.shape[0] - no_defect_count}")

# Visualize the distribution of each defect type
plt.figure(figsize=(12, 6))
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title('Distribution of Defect Types (Multi-Label)')
plt.xlabel('Defect Type')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=45)
plt.tight_layout()
# --- Add savefig before show ---
if eda_folder:
    try:
        plot_filename_dist = os.path.join(eda_folder, 'defect_type_distribution.png')
        plt.savefig(plot_filename_dist, bbox_inches='tight') # bbox_inches='tight' helps prevent labels being cut off
        print(f"Saved target distribution plot to: {plot_filename_dist}")
    except Exception as e:
        print(f"Could not save target distribution plot: {e}")
# ---
plt.show()
# Observation: This shows the frequency of each defect. Expect potential imbalance.
# Note that one steel plate can have multiple defects, so the sum of counts
# can be greater than the number of samples with defects.

# 2.e: Visualizations for Features (`2.3 Bons graficos...`)

# Histograms for numerical features
print("\n--- 2.e Feature Visualizations ---")
print("Generating histograms for numerical features...")
# Select a subset of features or plot all if manageable
num_features_to_plot = len(feature_columns) # Adjust if too many features
# Determine grid size for subplots
n_cols = 5
n_rows = (num_features_to_plot + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes = axes.flatten() # Flatten to 1D array for easy indexing

for i, col in enumerate(feature_columns[:num_features_to_plot]):
    if pd.api.types.is_numeric_dtype(df_train[col]): # Only plot numeric types
        sns.histplot(df_train[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    else:
        axes[i].set_title(f'{col} (Non-Numeric)', fontsize=10)
        axes[i].set_xticks([]) # Hide axis for non-numeric
        axes[i].set_yticks([])

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Histograms of Numerical Features', y=1.02, fontsize=16)
# --- Add savefig before show ---
if eda_folder:
    try:
        plot_filename_hist = os.path.join(eda_folder, 'feature_histograms.png')
        plt.savefig(plot_filename_hist, bbox_inches='tight')
        print(f"Saved feature histograms plot to: {plot_filename_hist}")
    except Exception as e:
        print(f"Could not save feature histograms plot: {e}")
# ---
plt.show()
# Observation: Check for skewness, multimodality, and range of values. This reinforces
# the need (or lack thereof) for normalization/standardization.

# Correlation Heatmap (`2.3 ...relações entre variaveis e com a target var`)
print("\nGenerating correlation heatmaps...")
# Calculate correlation matrix (including targets for feature-target correlation)
# Ensure only numeric columns are used for correlation
numeric_feature_columns = df_train[feature_columns].select_dtypes(include=np.number).columns.tolist()
if len(numeric_feature_columns) < len(feature_columns):
    print(f"Warning: Excluded {len(feature_columns)-len(numeric_feature_columns)} non-numeric columns from correlation analysis.")

correlation_matrix = df_train[numeric_feature_columns + target_columns].corr()

# Select correlations of features with target variables
if not correlation_matrix.empty and target_columns:
    feature_target_corr = correlation_matrix[target_columns].loc[numeric_feature_columns]

    # Visualize the feature-target correlations
    print("\nGenerating feature-target correlation heatmap...")
    plt.figure(figsize=(10, max(8, len(numeric_feature_columns) * 0.3))) # Adjust height based on number of features
    sns.heatmap(feature_target_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', annot_kws={"size": 8})
    plt.title('Correlation between Numeric Features and Defect Types (Targets)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # --- Add savefig before show ---
    if eda_folder:
        try:
            plot_filename_corr_target = os.path.join(eda_folder, 'feature_target_correlation_heatmap.png')
            plt.savefig(plot_filename_corr_target, bbox_inches='tight')
            print(f"Saved feature-target correlation heatmap to: {plot_filename_corr_target}")
        except Exception as e:
            print(f"Could not save feature-target correlation heatmap: {e}")
    # ---
    plt.show()
else:
    print("Skipping feature-target correlation heatmap (no numeric features or targets found).")


# Visualize the correlation between numeric features themselves (optional but good)
if not correlation_matrix.empty and len(numeric_feature_columns) > 1:
    print("\nGenerating feature-feature correlation heatmap...")
    feature_feature_corr = correlation_matrix.loc[numeric_feature_columns, numeric_feature_columns]
    plt.figure(figsize=(18, 15)) # May need adjustment depending on number of features
    sns.heatmap(feature_feature_corr, annot=False, cmap='coolwarm', fmt=".1f") # Annot=False if too crowded
    plt.title('Correlation between Numerical Features')
    plt.tight_layout()
    # --- Add savefig before show ---
    if eda_folder:
        try:
            plot_filename_corr_features = os.path.join(eda_folder, 'feature_feature_correlation_heatmap.png')
            plt.savefig(plot_filename_corr_features, bbox_inches='tight')
            print(f"Saved feature-feature correlation heatmap to: {plot_filename_corr_features}")
        except Exception as e:
            print(f"Could not save feature-feature correlation heatmap: {e}")
    # ---
    plt.show()
else:
     print("Skipping feature-feature correlation heatmap (less than 2 numeric features).")

# Observation: High positive (red) or negative (blue) correlations between features
# might indicate multicollinearity (redundancy). Features highly correlated with
# target variables are potentially good predictors. Low correlation doesn't necessarily
# mean a feature is useless (non-linear relationships might exist).

# 2.f: Initial thoughts on Preprocessing (`2.2 Normalizar?`)
print("\n--- 2.f Initial Preprocessing Considerations (Based on EDA) ---")
# Based on `.describe()` and histograms:
stats = df_train[numeric_feature_columns].describe().T
if not stats.empty:
    # Check if standard deviation exists and is non-zero for min/max checks
    if 'std' in stats.columns and 'min' in stats.columns and 'max' in stats.columns:
        min_std = stats['std'].min()
        # Avoid division by zero or near-zero if min is negative or zero
        max_range_ratio = (stats['max'] / (stats['min'].abs() + 1e-6)).max()

        # Example criteria for suggesting normalization (adjust thresholds as needed)
        if min_std > 0.01 and max_range_ratio > 100 :
             print("- Feature scales vary significantly (based on std dev and max/min ranges).")
             print("  Consider Normalization (e.g., MinMaxScaler) or Standardization (e.g., StandardScaler)")
             print("  before applying distance-based or gradient-based models (k-NN, SVM, NN).")
        else:
             print("- Feature scales seem relatively comparable or normalization might not be strictly required, but still often beneficial.")
    else:
        print("- Could not fully assess feature scales due to missing stats columns (std, min, max).")
else:
    print("- No numeric features found to assess scales for normalization.")


if missing_values.sum() > 0:
    print("- Missing values detected. Need an imputation strategy (e.g., mean, median, mode) or row removal.")
else:
    print("- No missing values detected. No imputation needed based on this check.")

# Check for non-numeric feature columns that weren't targets or IDs
non_numeric_features = df_train[feature_columns].select_dtypes(exclude=np.number).columns.tolist()
if non_numeric_features:
    print(f"- Non-numeric feature columns found: {', '.join(non_numeric_features)}. These may require encoding (e.g., OneHotEncoder) if they are categorical.")
else:
    print("- Data types seem appropriate (mostly numerical features). No obvious categorical encoding needed unless identified otherwise.")

print("- Target variables are multi-label binary indicators.")
if not target_counts.empty:
    min_target_count = target_counts.min()
    max_target_count = target_counts.max()
    if max_target_count / (min_target_count + 1e-6) > 5: # Example threshold for imbalance
        print("- Class imbalance observed in target variables. May need techniques like resampling (SMOTE, over/undersampling) or using appropriate metrics (Precision, Recall, F1, AUC) during modeling.")
    else:
        print("- Target variable distribution seems relatively balanced, but monitor metrics closely.")
else:
    print("- Could not assess target balance (no target counts).")


print("\n--- End of Initial EDA ---")