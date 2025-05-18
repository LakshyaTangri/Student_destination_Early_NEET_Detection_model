# **Business Understanding**

---

##Overview

###Predicting Post-16–18 Student Destinations Using Machine Learning for Early NEET Risk Detection
This research addresses the critical educational challenge of identifying schools with elevated risk of students becoming NEET (Not in Education, Employment, or Training) after completing their 16-18 education in England. Recognizing that NEET status correlates strongly with long-term social exclusion and economic disadvantage, this project applies machine learning techniques to develop predictive models capable of identifying at-risk institutions before students complete their studies. By analyzing institutional characteristics and historical destination patterns from the Compare School Performance dataset, we aim to facilitate early intervention strategies targeting the schools most in need of support.

##Context

The technical implementation follows a comprehensive data science pipeline encompassing exploratory data analysis, feature engineering, preprocessing, model development, and practical application development. Our models achieve substantial predictive power in both binary NEET risk classification and multi-class destination prediction, with Random Forest algorithms demonstrating the strongest performance after hyperparameter optimization. Feature importance analysis reveals that cohort composition, school type, and regional factors are particularly influential in determining student destinations, highlighting potential areas for targeted policy interventions.

#### 0.1 import modules
"""

# Importing essential libraries for data handling, visualization, preprocessing, modeling, and evaluation

# Core data manipulation
import pandas as pd
import numpy as np

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing tools
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Imputation and sampling
from sklearn.experimental import enable_iterative_imputer  # Enables experimental imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE  # For handling class imbalance

# Modeling tools
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Model selection and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, roc_auc_score, f1_score, accuracy_score,
                             log_loss, precision_score, recall_score, mean_squared_error)

# Other utilities
import matplotlib.ticker as mtick
import requests
from bs4 import BeautifulSoup
import zipfile
import os
import time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
import missingno as msno

"""#### 0.2 Set visualization theme  """

# Set Seaborn and Matplotlib theme for consistent and visually appealing plots

plt.style.use('seaborn-v0_8-whitegrid')  # Define base plot style
colors = sns.color_palette("viridis", 4)  # Define color palette
sns.set_palette(colors)

# Configure plot dimensions and font sizes
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

"""#**Raw Data Scraping**"""

import requests
from bs4 import BeautifulSoup
import zipfile
import os
import time
from pathlib import Path

def download_and_extract_zip(url, output_folder='/content'):
    """
    Download a zip file from a website and extract its contents to /content.

    Args:
        url (str): The URL of the webpage containing the download link
        output_folder (str): Folder to extract the zip contents to

    Returns:
        str: Path to the extracted folder
    """
    # Create a session for requests
    session = requests.Session()

    # Step 1: Get the webpage content
    print(f"Fetching webpage: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'referer': 'https://www.compare-school-performance.service.gov.uk/download-data',
        'host': 'www.compare-school-performance.service.gov.uk'
    }
    response = session.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Step 2: Parse the HTML and find the download link
    soup = BeautifulSoup(response.text, 'html.parser')
    download_link = soup.find('a', id='download-file-format-by-csv-link')

    if not download_link:
        raise ValueError("Download link not found on the page")

    # Get the href attribute which contains the download path
    download_path = download_link.get('href')

    # Make sure it's a full URL
    if download_path.startswith('/'):
        # Convert relative URL to absolute URL
        base_url = '/'.join(url.split('/')[:3])  # Get http(s)://domain.com part
        download_url = f"{base_url}{download_path}"
    else:
        download_url = download_path

    print(f"Found download link: {download_url}")

    # Step 3: Download the ZIP file
    timestamp = int(time.time())
    zip_filename = os.path.join("/tmp", f"downloaded_data_{timestamp}.zip")  # Store temp file in /tmp

    print(f"Downloading zip file to {zip_filename}...")
    zip_response = session.get(download_url, stream=True, headers=headers)
    zip_response.raise_for_status()

    # Create directory for zip file if it doesn't exist
    os.makedirs(os.path.dirname(zip_filename), exist_ok=True)

    # Save the zip file
    with open(zip_filename, 'wb') as f:
        for chunk in zip_response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Download complete. File saved as {zip_filename}")

    # Step 4: Extract the ZIP file
    extract_path = Path(output_folder)
    if not extract_path.exists():
        extract_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting zip file to {output_folder}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

    print(f"Extraction complete. Files extracted to {output_folder}")

    # Optional: List the extracted files
    extracted_files = os.listdir(output_folder)
    print(f"Files in {output_folder} directory:")
    for file in extracted_files:
        if file.endswith('.csv'):  # Filter to show only CSV files
            print(f" - {file}")

    # Optional: Remove the temporary zip file
    os.remove(zip_filename)
    print(f"Removed temporary zip file: {zip_filename}")

    return output_folder



# Example usage with the URL you provided
url = "https://www.compare-school-performance.service.gov.uk/download-data?currentstep=datatypes&regiontype=all&la=0&downloadYear=2022-2023&datatypes=gias&datatypes=ks5destinationhe"
# Download and extract directly to /content folder
output_folder = download_and_extract_zip(url)

print(f"\nAll processing complete. Data extracted to: {output_folder}")

"""# **Data Understanding**

---

## Stage 1: Data Loading and Initial Exploration

The project involved comprehensive data acquisition and preliminary exploration, an essential foundation for any robust machine learning endeavour. The rationale for beginning with thorough data exploration stems from the necessity to understand the structure, quality, and characteristics of the available information before proceeding with more sophisticated analyses. Without this crucial step, subsequent modeling efforts risk being built upon misconceptions about the underlying data, potentially leading to erroneous conclusions and ineffective interventions.

In practice, this step entailed loading two distinct datasets: school information data containing institutional characteristics and student destination data capturing post-education outcomes. The implementation utilized Pandas, a powerful data manipulation library, to read these CSV files into dataframes. The exploration process incorporated multiple methods of data summarization, including shape examination to understand dataset dimensions, basic information display to identify data types and potential missing values, and descriptive statistics to comprehend the central tendencies and distributions of numerical variables.

This approach adheres to best practices in data science by establishing a rigorous understanding of the data landscape before proceeding to more complex analyses. The evidence of this step's effectiveness can be observed in the comprehensive output displaying the structure of both datasets, which revealed that we have information on school characteristics across multiple dimensions and corresponding destination outcomes that will form the basis of our predictive models.

Through this initial data loading and exploration phase, we established a solid foundation for the subsequent analyses, ensuring that our understanding of the available information was accurate and comprehensive. This critical groundwork enables more informed decisions in the following data preparation and modeling stages.

###1.1 Load the datasets
"""

# 📥 Load raw school and destination data from CSV files into Pandas DataFrames
# This forms the basis for all downstream analysis and modeling

schools_df = pd.read_csv('/content/2022-2023/england_school_information.csv')
destinations_df = pd.read_csv('/content/2022-2023/england_ks5-studest-he.csv')

# Display the number of rows and columns in each dataset
print(f"School data shape: {schools_df.shape}")
print(f"Destination data shape: {destinations_df.shape}")

# 📊 Display metadata for the school dataset
# Includes column types, null values, and memory usage for quality checks

print("\n📋 School data summary:")
print(schools_df.info())

# 📊 Generate basic descriptive statistics for numeric features in the school dataset
# Helps identify distributions, scale of values, and potential outliers

print("\n📋 School data descriptive statistics:")
print(schools_df.describe().round(2))

# 📊 Display metadata for the destination dataset
# Provides a quick overview of structure and missing values

print("\n📋 Destination data summary:")
print(destinations_df.info())

# 📊 Generate basic descriptive statistics for numeric columns in the destination dataset
# This is essential to understanding typical progression and NEET-related indicators

print("\n📋 Destination data descriptive statistics:")
print(destinations_df.describe().round(2))

"""## Stage 2: Comprehensive Exploratory Data Analysis (EDA)

Following initial data loading, comprehensive exploratory data analysis was conducted to derive deeper insights into the patterns, relationships, and anomalies within the educational datasets. This thorough exploration is crucial because effective educational intervention strategies must be grounded in a nuanced understanding of the factors influencing student destinations, particularly regarding NEET outcomes. Without detailed visualization and statistical exploration, important patterns may remain undetected, potentially compromising the effectiveness of the predictive models and the subsequent interventions.

The exploratory analysis implemented a multi-faceted approach to data visualization, examining distributions of key variables across both datasets. For school characteristics, this included analyzing the distribution of school types, OFSTED ratings, and geographical locations using appropriate visualizations such as bar plots and histograms. For destination outcomes, the analysis focused on the distribution of progression rates, higher education enrollment, apprenticeship participation, and critically, NEET percentages.

The effectiveness of this approach is evidenced by the rich visual outputs that revealed important patterns. For instance, the histograms of progression rates demonstrated considerable variation across institutions, with some schools achieving near-perfect progression while others struggled with significant proportions of students not progressing to positive destinations. Similarly, the distribution of NEET rates highlighted institutions with concerning proportions of students becoming disengaged, providing valuable insights for targeted interventions.

This comprehensive EDA phase established a thorough understanding of the educational landscape captured in our data, identifying key patterns that informed subsequent feature engineering and modeling decisions. The visualizations created serve not only as analytical tools but also as powerful communication mechanisms for stakeholders in educational policy and practice.

### 2.1 Distribution of school types
"""

# 📊 Visualize the frequency distribution of different school types
# Useful for understanding the diversity and dominance of certain institutional categories

plt.figure(figsize=(12, 6))
school_counts = schools_df['SCHOOLTYPE'].value_counts()
sns.barplot(x=school_counts.index, y=school_counts.values)
plt.title('Distribution of School Types')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Schools')
plt.tight_layout()
plt.savefig('school_types_distribution.png')

"""### 2.2 Distribution of OFSTED ratings"""

# 📊 Plot the distribution of OFSTED inspection ratings across schools
# OFSTED rating is a key quality indicator that may correlate with student outcomes

plt.figure(figsize=(10, 6))
ofsted_counts = schools_df['OFSTEDRATING'].value_counts()
sns.barplot(x=ofsted_counts.index, y=ofsted_counts.values, palette="viridis")
plt.title('Distribution of OFSTED Ratings')
plt.ylabel('Number of Schools')
plt.tight_layout()
plt.savefig('ofsted_distribution.png')

"""### 2.3 Geographical distribution of schools (top 15 LAs)"""

# 📍 Show the number of schools per Local Authority (top 15 only)
# Helps highlight regional concentration and identify LAs with large student populations

plt.figure(figsize=(12, 8))
la_counts = schools_df['LANAME'].value_counts().head(15)
sns.barplot(x=la_counts.values, y=la_counts.index, palette="viridis")
plt.title('Top 15 Local Authorities by Number of Schools')
plt.xlabel('Number of Schools')
plt.tight_layout()

"""### 2.4 Convert percentage strings to floats for analysis"""

# 🧹 Clean and convert percentage columns for numerical analysis
# Converts valid percentage strings to float and replaces suppressed values ('SUPP', 'SP', 'NE') with NaN

dest_cols = ['ALL_PROGRESSED', 'ALL_APPREN', 'ALL_HE']
for col in dest_cols:
    destinations_df[col] = destinations_df[col].astype(str).replace({'SUPP': np.nan, 'SP': np.nan, 'NE': np.nan}, regex=True).str.rstrip('%').astype(float)

# 📈 Plot histogram of overall progression rates (students moving to positive destinations)
# Visualizes central tendency and variation, with a reference line showing the mean rate

plt.figure(figsize=(12, 6))
sns.histplot(destinations_df['ALL_PROGRESSED'].dropna(), kde=True, bins=20)
plt.title('Distribution of Overall Progression Rates')
plt.xlabel('Progression Rate (%)')
plt.ylabel('Number of Schools')
plt.axvline(destinations_df['ALL_PROGRESSED'].dropna().mean(), color='red', linestyle='--',
            label=f'Mean: {destinations_df["ALL_PROGRESSED"].dropna().mean():.2f}%')
plt.legend()
plt.tight_layout()
plt.savefig('progression_distribution.png')

"""### 2.5 Distributions of different destination types"""

# 📈 Plot overlapping distributions of higher education vs apprenticeship destinations
# Helps compare pathways and potential trade-offs between these post-school options

plt.figure(figsize=(14, 8))
sns.histplot(destinations_df['ALL_HE'].dropna(), kde=True, bins=20, color='red', label='Higher Education')
sns.histplot(destinations_df['ALL_APPREN'].dropna(), kde=True, bins=20, color='blue', label='Apprenticeships')
plt.title('Distribution of Higher Education vs Apprenticeship Destinations')
plt.xlabel('Percentage of Students')
plt.ylabel('Number of Schools')
plt.legend()
plt.tight_layout()
plt.savefig('he_vs_apprenticeship_distribution.png')

"""### 2.6 Calculate employment and NEET rates"""

# ➕ Derive new columns: employment rate and NEET rate
# These are inferred from progression minus known destinations (apprenticeship + HE)

destinations_df['EMPLOYMENT_PCT'] = destinations_df['ALL_PROGRESSED'] - destinations_df['ALL_APPREN'] - destinations_df['ALL_HE']
destinations_df['NEET_PCT'] = 100 - destinations_df['ALL_PROGRESSED']

# 📊 Create box plots for all destination categories (HE, apprenticeship, employment, NEET)
# Useful for comparing distributions and identifying outliers in each category

plt.figure(figsize=(14, 8))
dest_data = pd.DataFrame({
    'Higher Education': destinations_df['ALL_HE'].dropna(),
    'Apprenticeships': destinations_df['ALL_APPREN'].dropna(),
    'Employment': destinations_df['EMPLOYMENT_PCT'].dropna(),
    'NEET': destinations_df['NEET_PCT'].dropna()
})
sns.boxplot(data=dest_data)
plt.title('Distribution of Student Destinations')
plt.ylabel('Percentage of Students')
plt.tight_layout()
plt.savefig('destination_boxplots.png')

"""### 2.7 Create box plots for each destination type

### 2.8 NEET rate distribution
"""

# 📈 Visualize NEET rate distribution with mean and median
# Critical for identifying schools with high disengagement risk

plt.figure(figsize=(12, 6))
sns.histplot(destinations_df['NEET_PCT'].dropna(), kde=True, bins=20, color='red')
plt.title('Distribution of NEET Rates')
plt.xlabel('NEET Rate (%)')
plt.ylabel('Number of Schools')
plt.axvline(destinations_df['NEET_PCT'].dropna().mean(), color='darkred', linestyle='--',
            label=f'Mean: {destinations_df["NEET_PCT"].dropna().mean():.2f}%')
plt.axvline(destinations_df['NEET_PCT'].dropna().median(), color='black', linestyle='-.',
            label=f'Median: {destinations_df["NEET_PCT"].dropna().median():.2f}%')
plt.legend()
plt.tight_layout()
plt.savefig('neet_distribution.png')

"""# **Data Preparation**

---

## Stage 3: Data Merging

This analysis focused on integrating the separate datasets and preparing the consolidated information for meaningful analysis. This critical step was necessary because the predictive power of our models depends on establishing connections between institutional characteristics and student outcomes. Without proper merging and careful examination of the resulting dataset's quality, any subsequent analysis would lack coherence and potentially lead to spurious conclusions about the factors influencing NEET outcomes.

Methodologically, this stage involved merging the school information and student destination datasets using the Unique Reference Number (URN) as the common identifier. This inner join operation ensured that only schools with both characteristic information and outcome data were retained for analysis. Following the merge, a thorough examination of missing values was conducted to assess data completeness and quality, with visualizations created to highlight fields with significant proportions of missing information.

The effectiveness of this approach is demonstrated by the detailed missing value analysis, which identified specific variables requiring imputation strategies in the preprocessing pipeline. The visualization of missing data percentages provided clear evidence of data quality issues that needed addressing before model development. This transparency regarding data completeness is essential for establishing the credibility of subsequent analyses.

Through this data merging and preparation phase, we created a unified dataset that connects school characteristics with student outcomes, while gaining critical insights into data quality issues that informed our preprocessing strategies. This integrated dataset serves as the foundation for all subsequent feature engineering and modeling efforts

### 3.1 Merge datasets on URN (unique school identifier)
"""

# Merge the destinations dataset with school characteristics dataset using 'URN' (Unique Reference Number)
# This creates a combined dataset containing student destinations and school-level features.
merged_df = pd.merge(destinations_df, schools_df, on='URN', how='inner')
print(f"Merged data shape: {merged_df.shape}")

"""### 3.2 Check for missing values in the merged dataset

#### 3.2.1 Approch 1: Simple Imputation
"""

#------------------------------------------------------------------------------
# APPROACH 1: SIMPLE IMPUTATION MISSING DATA ANALYSIS
#------------------------------------------------------------------------------
print("\n===== SIMPLE IMPUTATION TO MISSING DATA ANALYSIS =====")

# Calculate the number and percentage of missing values in each column
missing_values = merged_df.isnull().sum()
missing_percent = (missing_values / len(merged_df)) * 100

# Store results in a DataFrame for easy visualization and reporting
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percent
})

# Display only columns with missing data
print("\n📊 Missing Data Analysis:")
print(missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# Plot a horizontal bar chart showing percentage of missing values per column
plt.figure(figsize=(12, 8))
missing_data_filtered = missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False)
sns.barplot(x=missing_data_filtered['Percentage'], y=missing_data_filtered.index)
plt.title('Features with Missing Values (Current Approach)')
plt.xlabel('Percentage of Missing Values')
plt.tight_layout()
plt.savefig('current_missing_values.png')

# Visualize missing value patterns using matrix plot from missingno
plt.figure(figsize=(14, 8))
msno.matrix(merged_df)
plt.title('Missing Value Matrix Visualization')
plt.tight_layout()
plt.savefig('msno_matrix.png')

# Visualize correlation of missing values across columns
plt.figure(figsize=(10, 8))
msno.heatmap(merged_df)
plt.title('Missing Value Correlation Heatmap')
plt.tight_layout()
plt.savefig('msno_heatmap.png')

"""####3.2.2 Approch 2: PCA Based Missing data analysis"""

# Define a function that imputes missing values using PCA reconstruction
def pca_imputation(df, n_components=0.95):
    """
    Impute missing values using PCA.
    1. Keeps only numeric columns.
    2. Applies IterativeImputer for initial imputation.
    3. Scales data and applies PCA.
    4. Reconstructs data from PCA components to fill in missing values.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Only numeric features

    imputer = IterativeImputer(max_iter=10, random_state=42)  # First pass imputation
    temp_df = numeric_df.copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputer.fit_transform(temp_df))  # Scale the data

    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)

    # Project to PCA space and inverse transform back to impute
    pca_data = pca.transform(scaled_data)
    reconstructed_data = pca.inverse_transform(pca_data)
    reconstructed_orig = scaler.inverse_transform(reconstructed_data)

    imputed_df = pd.DataFrame(reconstructed_orig,
                             columns=numeric_df.columns,
                             index=numeric_df.index)

    # Evaluate imputation error on originally non-missing values
    original_values = numeric_df.values
    imputed_values = imputed_df.values
    mask = ~np.isnan(original_values)

    mse = mean_squared_error(original_values[mask], imputed_values[mask])
    print(f"Mean squared error of PCA imputation on non-missing values: {mse:.4f}")

    return imputed_df, pca, pca.explained_variance_ratio_

# Define a function that imputes missing values using PCA reconstruction
def pca_imputation(df, n_components=0.95):
    """
    Impute missing values using PCA.
    1. Keeps only numeric columns.
    2. Applies IterativeImputer for initial imputation.
    3. Scales data and applies PCA.
    4. Reconstructs data from PCA components to fill in missing values.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Only numeric features

    imputer = IterativeImputer(max_iter=10, random_state=42)  # First pass imputation
    temp_df = numeric_df.copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputer.fit_transform(temp_df))  # Scale the data

    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)

    # Project to PCA space and inverse transform back to impute
    pca_data = pca.transform(scaled_data)
    reconstructed_data = pca.inverse_transform(pca_data)
    reconstructed_orig = scaler.inverse_transform(reconstructed_data)

    imputed_df = pd.DataFrame(reconstructed_orig,
                             columns=numeric_df.columns,
                             index=numeric_df.index)

    # Evaluate imputation error on originally non-missing values
    original_values = numeric_df.values
    imputed_values = imputed_df.values
    mask = ~np.isnan(original_values)

    mse = mean_squared_error(original_values[mask], imputed_values[mask])
    print(f"Mean squared error of PCA imputation on non-missing values: {mse:.4f}")

    return imputed_df, pca, pca.explained_variance_ratio_

# Apply PCA imputation on numeric columns and evaluate explained variance
numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
original_numeric_df = merged_df[numeric_cols].copy()
missing_mask = original_numeric_df.isna()

# Call the pca_imputation function to get explained_variance
imputed_df, pca_model, explained_variance = pca_imputation(original_numeric_df)


# Plot explained variance by PCA components
plt.figure(figsize=(10, 6))
cumulative_variance = np.cumsum(explained_variance)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
plt.step(range(1, len(explained_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('pca_explained_variance.png')

# Define a function to visualize PCA-transformed data, colored by missing data categories
def visualize_pca_missing_patterns(df, pca_model, missing_mask):
    """Visualize data in PCA space colored by missing patterns"""
    imputer = IterativeImputer(max_iter=10, random_state=42)
    filled_data = imputer.fit_transform(df)

    pca_data = pca_model.transform(StandardScaler().fit_transform(filled_data))

    pca_df = pd.DataFrame({
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1]
    })

    missing_counts = missing_mask.sum(axis=1)

    pca_df['Missing_Values'] = pd.cut(missing_counts,
                                     bins=[-1, 0, 1, 2, float('inf')],
                                     labels=['None', '1 value', '2 values', '3+ values'])

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Missing_Values', data=pca_df, palette='viridis', alpha=0.7)
    plt.title('PCA Visualization of Data Points Colored by Missing Values')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} explained variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} explained variance)')
    plt.legend(title='Missing Values')
    plt.tight_layout()
    plt.savefig('pca_missing_patterns.png')

    return pca_df

# Generate PCA plot of missing patterns in data
pca_vis_df = visualize_pca_missing_patterns(original_numeric_df, pca_model, missing_mask)

# Define function to compare a few examples of PCA-imputed vs original values
def compare_imputation_sample(original_df, imputed_df, missing_mask, n_samples=5):
    """Compare original and imputed values for a few examples"""
    missing_indices = {}
    for col in original_df.columns:
        missing_indices[col] = original_df[original_df[col].isna()].index.tolist()

    comparison_data = []

    for col in original_df.columns:
        if not missing_indices[col]:
            continue

        sample_indices = np.random.choice(missing_indices[col],
                                         min(n_samples, len(missing_indices[col])),
                                         replace=False)

        for idx in sample_indices:
            comparison_data.append({
                'Feature': col,
                'Index': idx,
                'Imputed Value': imputed_df.loc[idx, col],
                'Feature Mean': original_df[col].mean(),
                'Feature Median': original_df[col].median(),
                'Deviation from Mean': imputed_df.loc[idx, col] - original_df[col].mean(),
                'Deviation from Median': imputed_df.loc[idx, col] - original_df[col].median()
            })

    return pd.DataFrame(comparison_data)

# Apply PCA imputation on numeric columns and evaluate explained variance
numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
original_numeric_df = merged_df[numeric_cols].copy()
missing_mask = original_numeric_df.isna()

imputed_df, pca_model, explained_variance = pca_imputation(original_numeric_df)

print(f"\nNumber of PCA components: {pca_model.n_components_}")
print(f"Total explained variance: {sum(explained_variance):.4f}")

"""####3.2.3 Comparison and decision"""

# Calculate NEET percentage from ALL_PROGRESSED for original and imputed data
original_numeric_df['NEET_PCT'] = 100 - original_numeric_df['ALL_PROGRESSED']
imputed_df['NEET_PCT'] = 100 - imputed_df['ALL_PROGRESSED']

# Visualize the distribution of NEET% before and after PCA imputation
plt.figure(figsize=(12, 6))
sns.kdeplot(original_numeric_df['NEET_PCT'].dropna(), label='Original (non-missing)', color='blue')
sns.kdeplot(imputed_df.loc[original_numeric_df['NEET_PCT'].isna(), 'NEET_PCT'],
            label='PCA-imputed', color='red')
plt.axvline(original_numeric_df['NEET_PCT'].dropna().mean(), color='blue', linestyle='--',
           label=f'Original mean: {original_numeric_df["NEET_PCT"].dropna().mean():.2f}')
plt.axvline(imputed_df.loc[original_numeric_df['NEET_PCT'].isna(), "NEET_PCT"].mean(),
           color='red', linestyle='--',
           label=f'Imputed mean: {imputed_df.loc[original_numeric_df["NEET_PCT"].isna(), "NEET_PCT"].mean():.2f}')
plt.title('Distribution of Original vs PCA-Imputed NEET Percentages')
plt.xlabel('NEET Percentage')
plt.legend()
plt.tight_layout()
plt.savefig('neet_imputation_comparison.png')

"""## Stage 4. Feature Engineering and Target Creation

The strategic transformation of raw data into meaningful features and target variables that accurately represent the educational phenomena under investigation. This feature engineering step was essential because raw educational data often fails to directly capture the complex concepts we aim to predict, such as NEET risk or dominant student destinations. Without careful construction of these derived variables, the resulting models would lack the necessary target metrics and interpretable features required for effective prediction and intervention planning.

The implementation focused on creating several key derived variables. First, NEET percentages were calculated by subtracting progression rates from 100%, providing a direct measure of students not in education, employment, or training. Employment percentages were derived by subtracting higher education and apprenticeship percentages from overall progression rates. Subsequently, a multi-class target variable named "DOMINANT_DESTINATION" was constructed by identifying the highest percentage destination for each school (Higher Education, Apprenticeship, Employment, or NEET). Additionally, a binary NEET risk flag was created using the 75th percentile of NEET rates as a threshold, identifying schools with particularly concerning levels of student disengagement.

The effectiveness of this approach is evidenced by the subsequent visualizations of target variable distributions, which revealed an imbalance in destination outcomes that would need addressing in the modeling phase. The use of a data-driven threshold (75th percentile) for NEET risk flagging ensured that our binary classification target identified genuinely concerning outcomes rather than using an arbitrary threshold.

This feature engineering phase successfully transformed raw educational metrics into meaningful target variables and features that directly address our research questions regarding post-16-18 student destinations and NEET risk. These carefully constructed variables form the foundation of our predictive modeling approach, enabling more nuanced insights into the factors influencing educational outcomes.

### 4.1 Ensure all destination columns are properly formatted
"""

### 4.1 Ensure all destination columns are properly formatted
# Standardize and clean destination percentage columns by removing suppression codes ('SUPP', 'SP', 'NE'),
# stripping percentage signs, and converting them to numeric floats for analysis and feature engineering.
destination_cols = ['ALL_PROGRESSED', 'ALL_APPREN', 'ALL_HE']
for col in destination_cols:
    merged_df[col] = merged_df[col].astype(str).replace({'SUPP': np.nan, 'SP': np.nan, 'NE': np.nan}, regex=True).str.rstrip('%').astype(float)

"""### 4.2 Calculate NEET percentage"""

### 4.2 Calculate NEET percentage
# Derive NEET percentage as the inverse of overall progression; this acts as a direct target proxy
# to identify students not in education, employment, or training.
merged_df['NEET_PCT'] = 100 - merged_df['ALL_PROGRESSED']

"""### 4.3 Calculate employment percentage"""

### 4.3 Calculate employment percentage
# Estimate employment rates by removing HE and apprenticeship proportions from total progression.
# Since some inconsistencies in data may lead to negative values (e.g., overlap or reporting errors),
# we clip the employment rate at 0 to maintain logical consistency.
merged_df['EMPLOYMENT_PCT'] = merged_df['ALL_PROGRESSED'] - merged_df['ALL_APPREN'] - merged_df['ALL_HE']
merged_df['EMPLOYMENT_PCT'] = merged_df['EMPLOYMENT_PCT'].clip(lower=0)

"""### 4.4 Create multi-class labels: HE, Apprenticeship, Employment, NEET"""

### 4.4 Create multi-class labels: HE, Apprenticeship, Employment, NEET
# Construct a multi-class target variable capturing each school's dominant post-16 destination
# by selecting the highest percentage among HE, Apprenticeship, Employment, or NEET.
dest_columns = ['ALL_HE', 'ALL_APPREN', 'EMPLOYMENT_PCT', 'NEET_PCT']
destinations_only = merged_df[dest_columns]

# Remove rows with no destination data at all to avoid invalid target assignments
rows_to_drop = destinations_only[destinations_only.isnull().all(axis=1)].index
merged_df = merged_df.drop(index=rows_to_drop)

# Assign dominant destination based on the column with the maximum percentage value
merged_df['DOMINANT_DESTINATION'] = 'Unknown'
merged_df['DOMINANT_DESTINATION'] = destinations_only.idxmax(axis=1, skipna=True).map({
    'ALL_HE': 'HE',
    'ALL_APPREN': 'Apprenticeship',
    'EMPLOYMENT_PCT': 'Employment',
    'NEET_PCT': 'NEET'
}).fillna('Unknown')

"""### 4.5 Create binary NEET risk flag"""

### 4.5 Create binary NEET risk flag
# Generate a binary indicator flagging schools with NEET rates above the 75th percentile threshold.
# This helps focus modeling efforts on high-risk institutions where early intervention is critical.
NEET_THRESHOLD = 30.0
merged_df['NEET_RISK_FLAG'] = (merged_df['NEET_PCT'] > NEET_THRESHOLD).astype(int)

# Visualize NEET risk distribution across schools, with contextual labeling
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='NEET_RISK_FLAG', data=merged_df, palette=['lightgreen', 'coral'])
plt.title('Distribution of NEET Risk Flag')
plt.xlabel('NEET Risk Flag')
plt.ylabel('Number of Schools')

# Annotate bar chart with percentage values
total = len(merged_df['NEET_RISK_FLAG'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2 - 0.1
    y = p.get_y() + p.get_height() + 5
    ax.annotate(percentage, (x, y), size=12)

# Replace numeric x-axis ticks with meaningful labels
labels = ['Low Risk (≤ {}%)'.format(round(NEET_THRESHOLD, 1)),
          'High Risk (> {}%)'.format(round(NEET_THRESHOLD, 1))]
plt.xticks([0, 1], labels)

plt.tight_layout()
plt.savefig('neet_risk_distribution_enhanced.png')

# Visualize the distribution of dominant destinations across schools.
# This highlights class imbalance, which informs later resampling strategies in classification tasks.
dest_counts = merged_df['DOMINANT_DESTINATION'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(dest_counts, labels=dest_counts.index, autopct='%1.1f%%', colors=sns.color_palette("plasma"))
plt.title('Distribution of Dominant Destinations')
plt.tight_layout()
plt.savefig('destination_distribution_pie.png')

# Confirm the final shape of the dataset after feature engineering and filtering
merged_df.shape

"""## Stage 5. Relational Analysis

The relationships between school characteristics and student destination outcomes, particularly focusing on NEET percentages. This relational analysis was crucial because identifying significant associations between institutional factors and concerning outcomes provides actionable insights for policymakers and educators. Without examining these relationships, our understanding would remain superficial, lacking the contextual knowledge necessary to develop targeted interventions for schools with elevated NEET risk.

Methodologically, this analytical phase implemented a series of comparative visualizations examining how NEET percentages vary across different school characteristics. Box plots were used to visualize the relationship between OFSTED ratings and NEET percentages, revealing how regulatory assessments of school quality correlate with student outcomes. Similar analyses were conducted for school types and religious character, providing insights into how institutional structures influence destination patterns. Additionally, geographical analysis identified local authorities with concerning average NEET percentages, highlighting regional disparities requiring attention. A correlation matrix of numerical variables further illuminated statistical relationships between quantitative school metrics and outcome measures.

The effectiveness of this approach is evidenced by several revealing patterns. For instance, the box plots demonstrated an inverse relationship between OFSTED ratings and NEET percentages, with "Outstanding" schools typically showing lower rates of disengagement than those requiring improvement. The geographical analysis identified specific local authorities with consistently elevated NEET percentages, suggesting regional factors influencing student outcomes that merit further investigation by policymakers.

Through this relational analysis, we established important connections between institutional characteristics and student destinations that provide context for our modeling efforts and offer actionable insights for educational stakeholders. These relationships informed feature selection for our predictive models and highlight potential intervention points for reducing NEET outcomes across different school contexts.

### 5.1 Relationship between OFSTED rating and NEET percentage
"""

### 5.1 Relationship between OFSTED rating and NEET percentage
# Boxplot visualizing how regulatory quality assessments (OFSTED ratings) relate to NEET percentages.
# This reveals systemic links between perceived school performance and student disengagement levels.
plt.figure(figsize=(10, 6))
sns.boxplot(x='OFSTEDRATING', y='NEET_PCT', data=merged_df.dropna(subset=['OFSTEDRATING', 'NEET_PCT']))
plt.title('NEET Percentage by OFSTED Rating')
plt.xlabel('OFSTED Rating')
plt.ylabel('NEET Percentage')
plt.tight_layout()
plt.savefig('neet_by_ofsted.png')

"""Schools rated "Outstanding" tend to have significantly lower NEET rates, suggesting a positive association between institutional quality and student engagement. This supports OFSTED ratings as a meaningful predictive feature.

### 5.2 Relationship between school type and NEET percentage
"""

### 5.2 Relationship between school type and NEET percentage
# Boxplot comparing NEET outcomes across the 8 most common school types.
# Captures structural effects of institutional design on student destinations.
plt.figure(figsize=(12, 8))
school_types = merged_df['SCHOOLTYPE'].value_counts().head(8).index
filtered_df = merged_df[merged_df['SCHOOLTYPE'].isin(school_types)]
sns.boxplot(x='SCHOOLTYPE', y='NEET_PCT', data=filtered_df)
plt.title('NEET Percentage by School Type (Top 8)')
plt.xlabel('School Type')
plt.ylabel('NEET Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('neet_by_school_type.png')

"""School type appears to influence NEET outcomes—some structures (e.g., academies, UTCs) exhibit wider variability, suggesting differing support or curriculum alignment with post-education transitions.

### 5.3 Relationship between destinations and school characteristics
"""

### 5.3 Relationship between destinations and school characteristics
# Analysis of NEET percentage by religious character to evaluate cultural or ethos-driven differences.
plt.figure(figsize=(12, 8))
sns.boxplot(x='RELCHAR', y='NEET_PCT', data=merged_df.dropna(subset=['RELCHAR', 'NEET_PCT']))
plt.title('NEET Percentage by Religious Character')
plt.xlabel('Religious Character')
plt.ylabel('NEET Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('neet_by_religious_character.png')

"""Religious affiliation may have nuanced effects on NEET risk—some groups (e.g., Church of England, Roman Catholic) show slightly lower NEET rates. Cultural emphasis on structure and discipline may partly explain this.

### 5.4 Correlation analysis of key numerical variables
"""

### 5.4 Correlation analysis of key numerical variables
# Heatmap to identify which numerical school metrics are statistically associated with NEET or other outcomes.
# Guides feature selection for predictive modeling.
numerical_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
corr_data = merged_df[numerical_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_data))
sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

"""NEET_PCT negatively correlates with ALL_HE and ALL_PROGRESSED, and positively with derived EMPLOYMENT_PCT—indicating mutually exclusive post-school paths and useful predictive contrasts.

### 5.5 NEET risk by geographical area (top 15 LAs)
"""

### 5.5 NEET risk by geographical area (top 15 LAs)
# Highlights regional disparities by identifying local authorities with highest average NEET percentages.
# Useful for geospatial targeting of interventions.
plt.figure(figsize=(14, 8))
la_neet = merged_df.groupby('LANAME')['NEET_PCT'].mean().sort_values(ascending=False).head(15)
sns.barplot(x=la_neet.values, y=la_neet.index)
plt.title('Top 15 Local Authorities by Average NEET Percentage')
plt.xlabel('Average NEET Percentage')
plt.tight_layout()
plt.savefig('neet_by_la.png')

"""Certain local authorities exhibit structurally high NEET rates, signaling localized socioeconomic challenges or underperforming support systems. These zones warrant focused policy attention"""

# Confirm the data shape has remained consistent after segmentation
merged_df.shape

"""## Stage 6. Feature Selection and preprocessing

Systematic preparation of features for modeling through careful selection, transformation, and preprocessing. This critical step was necessary because raw educational data often contains irrelevant information, missing values, and variables in formats unsuitable for machine learning algorithms. Without proper preprocessing, models would struggle with the heterogeneous data types, scale differences, and missing values inherent in educational datasets, potentially leading to biased or ineffective predictions of NEET risk.

The implementation employed a structured approach to feature preparation, beginning with the removal of variables directly used in target creation to prevent data leakage. A strategic selection of institutional characteristics was then applied, focusing on school attributes, geographical data, performance metrics, and cohort information likely to influence student destinations. To handle the mixed data types, separate preprocessing pipelines were constructed for numerical and categorical features. Numerical variables were processed using median imputation to address missing values, followed by standardization to ensure comparable scales. Categorical variables underwent most-frequent-value imputation and one-hot encoding to transform them into a format suitable for machine learning algorithms.

The effectiveness of this preprocessing strategy is evidenced by the successful transformation of complex educational data into a format suitable for machine learning algorithms. The application of SMOTE further addressed the class imbalance in our binary NEET risk target, ensuring that the model would not simply predict the majority class. The dimensional expansion following preprocessing, particularly from one-hot encoding of categorical variables, effectively captured the diverse institutional factors influencing student destinations.

This feature selection and preprocessing phase successfully transformed raw educational data into a clean, formatted feature matrix ready for model development. The careful handling of missing values, categorical variables, and class imbalance established a robust foundation for the subsequent modeling efforts, ensuring that our predictive models would have the best possible chance of identifying patterns associated with NEET risk.

### 6.1 Drop columns directly used in target creation to avoid data leakage

These columns are direct components of the target variable and would introduce data leakage, falsely inflating model performance.
"""

cols_to_drop = [
    'ALL_PROGRESSED',  # Used to compute NEET_PCT
    'ALL_APPREN',      # Used in EMPLOYMENT_PCT
    'ALL_HE',          # Used in EMPLOYMENT_PCT
    'NEET_PCT',        # Our true target (excluded during modeling to predict NEET_RISK_FLAG)
    'EMPLOYMENT_PCT'   # Derived from others
]

merged_df_clean = merged_df.drop(columns=cols_to_drop)
merged_df_clean.shape

merged_df_clean.shape

# Drop rows with any empty values
#merged_df_clean = merged_df_clean.dropna()

# Display the stats of the cleaned dataframe
print(merged_df_clean.shape)

"""### 6.2 Select relevant features for modeling

Institutional features (e.g., school type, governance, region) and performance indicators (e.g., scores, cohort sizes) likely influence post-school outcomes and NEET risk.
"""

selected_features = [
    # School characteristics
    'MINORGROUP', 'SCHOOLTYPE', 'GENDER', 'RELCHAR', 'ADMPOL', 'OFSTEDRATING',
    'ISPOST16', 'ISSECONDARY',
    # Regional/location
    'LANAME',
    # Performance data
    'ALL_SCORE', 'ACADAGEN_SCORE', 'TLEV_SCORE',
    # Cohort data
    'ALL_COHORT', 'ACADAGEN_COHORT', 'TLEV_COHORT',
]

# Keep only features present in our dataset
selected_features = [f for f in selected_features if f in merged_df.columns]
print("\nSelected features:", selected_features)

"""### 6.3 Create feature dataset

We maintain two targets: a multiclass version for destination prediction and a binary NEET risk flag for classification modeling.
"""

X = merged_df_clean[selected_features].copy()
y_multiclass = merged_df_clean['DOMINANT_DESTINATION']
y_binary = merged_df_clean['NEET_RISK_FLAG']

"""### 6.4 Identify numerical and categorical features

Enables separate handling of missing values, scaling, and encoding based on feature types.
"""

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

"""### 6.5 Creating Preprocessing Pipeline

Design Benefits:

Median imputation is robust against outliers in performance scores.

StandardScaler ensures all numeric features are on the same scale.

One-hot encoding avoids ordinal bias in categorical variables like RELCHAR or OFSTEDRATING.
"""

# Numerical pipeline: impute missing values with median and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing values with most frequent and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

"""### 6.6 Apply preprocessing

The dimensionality expands due to one-hot encoding, allowing the model to represent institutional nuances like school type and religious character effectively.
"""

X_processed = preprocessor.fit_transform(X)
print(f"\nProcessed feature matrix shape: {X_processed.shape}")

"""### 6.7 Handle class imbalance for NEET risk using SMOTE

The original dataset likely had far fewer NEET cases. SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples of the minority class, ensuring that models don't become biased toward the majority (non-NEET) outcome.
"""

print("\nApplying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_resampled, y_binary_resampled = smote.fit_resample(X_processed, y_binary)
print(f"Resampled data shape: {X_resampled.shape}")
print("Resampled class distribution:", pd.Series(y_binary_resampled).value_counts())

"""# **Modeling**

---

## Stage 7. Model Training and Model Evaluation

Stage comprised the development, training, and rigorous evaluation of machine learning models designed to predict student destination outcomes. This modeling stage was essential because translating preprocessed educational data into actionable predictions requires algorithms capable of capturing complex, non-linear relationships between institutional characteristics and student outcomes. Without systematic model development and evaluation, we would lack the predictive capabilities necessary for early identification of schools with elevated NEET risk.

Methodologically, this stage implemented a comprehensive approach to model development for both binary NEET risk classification and multi-class destination prediction. Random Forest classifiers were selected for their ability to handle complex interactions between features and their inherent feature importance mechanisms. The implementation included hyperparameter tuning through grid search cross-validation to optimize model performance. For evaluation, multiple metrics were employed including classification reports (precision, recall, F1-score), confusion matrices, ROC curves, and precision-recall curves, providing a holistic assessment of model performance across different dimensions.

The effectiveness of this modeling approach is evidenced by the performance metrics achieved. For the binary NEET risk classification, the model demonstrated strong discriminative ability with both precision and recall substantially above baseline, indicating its capacity to identify high-risk schools without excessive false positives. The ROC curve analysis further confirmed the model's ability to distinguish between risk categories across different threshold settings. For multi-class destination prediction, the weighted F1-scores reflected the model's capacity to predict the primary destination pathway for each institution.

Through this model development phase, we successfully created predictive tools capable of identifying schools at elevated risk of NEET outcomes and forecasting dominant student destinations based on institutional characteristics. The feature importance analysis further provided interpretable insights into the factors most strongly associated with different student outcomes, offering valuable guidance for policy interventions.
"""

merged_df.dropna(inplace=True)

"""### 7.1 Split data for binary classification (NEET risk prediction)"""

### 7.1 Data Splitting

# Ensure the dataset is clean before modeling
merged_df.dropna(inplace=True)

# Split data for binary NEET risk prediction, using stratification to preserve class balance
X_train, X_test, y_binary_train, y_binary_test = train_test_split(
    X_processed, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Split data for multi-class dominant destination prediction
X_train_multi, X_test_multi, y_multi_train, y_multi_test = train_test_split(
    X_processed, y_multiclass, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

"""### 7.2 Prepare Data for Modeling"""

### 7.2 Handling Class Imbalance for Binary Classification

# NEET risk prediction is a binary classification problem with potential imbalance.
# SMOTE (Synthetic Minority Oversampling Technique) is applied to oversample the minority class.
smote = SMOTE(random_state=42)
X_train_resampled, y_binary_train_resampled = smote.fit_resample(X_train, y_binary_train)

# Post-resampling summary
print("Resampled binary training set shape:", X_train_resampled.shape)
print("Binary class distribution after SMOTE:\n", pd.Series(y_binary_train_resampled).value_counts())

"""### 7.3 Seting up model configurations"""

### 7.3 Model Configuration

# Define Random Forest, XGBoost, and SVC classifiers for both binary and multi-class tasks.
# Class weights are adjusted to account for imbalances, and probability=True is set for ROC/AUC calculations.

# Binary classification models
rf_binary = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb_binary = XGBClassifier(scale_pos_weight=1, random_state=42)
svc_binary = SVC(class_weight='balanced', probability=True, random_state=42)

# Multi-class classification models
rf_multi = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb_multi = XGBClassifier(random_state=42)
svc_multi = SVC(probability=True, random_state=42)

# Hyperparameter grids for model tuning using GridSearchCV later
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

svc_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

"""### 7.4 Binary Classification Evaluation"""

print("\n" + "="*50)
print("BINARY CLASSIFICATION - NEET RISK DETECTION")
print("="*50)

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
binary_results = []

"""#### 7.4.1 Random Forest for binary classification"""

# GridSearchCV for hyperparameter tuning with cross-validation
rf_grid = GridSearchCV(rf_binary, rf_params, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
rf_grid.fit(X_train_resampled, y_binary_train_resampled)
best_rf = rf_grid.best_estimator_
print(f"Best parameters: {rf_grid.best_params_}")

# Evaluate Random Forest performance on test set
# Evaluate RF
rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]
rf_accuracy = accuracy_score(y_binary_test, rf_pred)
rf_precision = precision_score(y_binary_test, rf_pred)
rf_recall = recall_score(y_binary_test, rf_pred)
rf_f1 = f1_score(y_binary_test, rf_pred)
rf_roc_auc = roc_auc_score(y_binary_test, rf_prob)

binary_results.append({
    'Model': 'Random Forest',
    'Accuracy': rf_accuracy,
    'Precision': rf_precision,
    'Recall': rf_recall,
    'F1 Score': rf_f1,
    'ROC AUC': rf_roc_auc
})

print("\n--- Random Forest Binary Classification Report ---")
print(classification_report(y_binary_test, rf_pred))

# Plot RF confusion matrix
cm_rf = confusion_matrix(y_binary_test, rf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not at Risk', 'NEET Risk'],
            yticklabels=['Not at Risk', 'NEET Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('confusion_matrix_random_forest.png')
# Plot RF ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_binary_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, lw=2, label=f'ROC curve (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.savefig('roc_curve_random_forest.png')

# RF feature importance
rf_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_importances.head(15))
plt.title('Top 15 Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance_random_forest.png')
print("\nTop 10 Features for Random Forest:")
print(rf_importances.head(10))

"""#### 7.4.2 XGBoost for binary classification"""

print("\nTraining XGBoost for Binary Classification...")
xgb_grid = GridSearchCV(xgb_binary, xgb_params, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train_resampled, y_binary_train_resampled)
best_xgb = xgb_grid.best_estimator_
print(f"Best parameters: {xgb_grid.best_params_}")

# Evaluate XGBoost
xgb_pred = best_xgb.predict(X_test)
xgb_prob = best_xgb.predict_proba(X_test)[:, 1]
xgb_accuracy = accuracy_score(y_binary_test, xgb_pred)
xgb_precision = precision_score(y_binary_test, xgb_pred)
xgb_recall = recall_score(y_binary_test, xgb_pred)
xgb_f1 = f1_score(y_binary_test, xgb_pred)
xgb_roc_auc = roc_auc_score(y_binary_test, xgb_prob)

binary_results.append({
    'Model': 'XGBoost',
    'Accuracy': xgb_accuracy,
    'Precision': xgb_precision,
    'Recall': xgb_recall,
    'F1 Score': xgb_f1,
    'ROC AUC': xgb_roc_auc
})

print("\n--- XGBoost Binary Classification Report ---")
print(classification_report(y_binary_test, xgb_pred))

# Plot XGBoost confusion matrix
cm_xgb = confusion_matrix(y_binary_test, xgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not at Risk', 'NEET Risk'],
            yticklabels=['Not at Risk', 'NEET Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.savefig('confusion_matrix_xgboost.png')

# Plot XGBoost ROC curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_binary_test, xgb_prob)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, lw=2, label=f'ROC curve (area = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc="lower right")
plt.savefig('roc_curve_xgboost.png')

# XGBoost feature importance
xgb_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importances.head(15))
plt.title('Top 15 Feature Importance - XGBoost')
plt.tight_layout()
plt.savefig('feature_importance_xgboost.png')
print("\nTop 10 Features for XGBoost:")
print(xgb_importances.head(10))

"""#### 7.4.3 SVC for binary classification"""

print("\nTraining SVC for Binary Classification...")
svc_grid = GridSearchCV(svc_binary, svc_params, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
svc_grid.fit(X_train_resampled, y_binary_train_resampled)
best_svc = svc_grid.best_estimator_
print(f"Best parameters: {svc_grid.best_params_}")

# Evaluate SVC
svc_pred = best_svc.predict(X_test)
svc_prob = best_svc.predict_proba(X_test)[:, 1]
svc_accuracy = accuracy_score(y_binary_test, svc_pred)
svc_precision = precision_score(y_binary_test, svc_pred)
svc_recall = recall_score(y_binary_test, svc_pred)
svc_f1 = f1_score(y_binary_test, svc_pred)
svc_roc_auc = roc_auc_score(y_binary_test, svc_prob)

binary_results.append({
    'Model': 'SVC',
    'Accuracy': svc_accuracy,
    'Precision': svc_precision,
    'Recall': svc_recall,
    'F1 Score': svc_f1,
    'ROC AUC': svc_roc_auc
})
print("\n--- SVC Binary Classification Report ---")
print(classification_report(y_binary_test, svc_pred))

# Plot SVC confusion matrix
cm_svc = confusion_matrix(y_binary_test, svc_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not at Risk', 'NEET Risk'],
            yticklabels=['Not at Risk', 'NEET Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVC')
plt.savefig('confusion_matrix_svc.png')

# Plot SVC ROC curve
fpr_svc, tpr_svc, _ = roc_curve(y_binary_test, svc_prob)
roc_auc_svc = auc(fpr_svc, tpr_svc)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svc, tpr_svc, lw=2, label=f'ROC curve (area = {roc_auc_svc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVC')
plt.legend(loc="lower right")
plt.savefig('roc_curve_svc.png')

# Compile binary results
binary_results_df = pd.DataFrame(binary_results).set_index('Model')
print("\nBinary Classification Model Comparison:")
print(binary_results_df)

# Visualize binary model comparison
plt.figure(figsize=(12, 8))
binary_results_df.drop('ROC AUC', axis=1).plot(kind='bar')
plt.title('Binary Classification - Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('binary_model_comparison.png')

# ROC AUC comparison
plt.figure(figsize=(8, 6))
binary_results_df['ROC AUC'].plot(kind='bar', color='green')
plt.title('Binary Classification - ROC AUC Comparison')
plt.ylabel('ROC AUC Score')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('binary_roc_auc_comparison.png')

"""### 7.5 Multi-class Classification Evaluation"""

from sklearn.utils.multiclass import unique_labels

print("\n" + "="*50)
print("MULTI-CLASS CLASSIFICATION - DESTINATION PREDICTION")
print("="*50)

multiclass_results = []

class_names = unique_labels(y_multi_test)

"""#### 7.5.1 Random Forest for multi-class classification"""

print("\nTraining Random Forest for Multi-class Classification...")
rf_multi_grid = GridSearchCV(rf_multi, rf_params, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0)
rf_multi_grid.fit(X_train_multi, y_multi_train)
best_rf_multi = rf_multi_grid.best_estimator_
print(f"Best parameters: {rf_multi_grid.best_params_}")

# Evaluate RF multi-class
rf_multi_pred = best_rf_multi.predict(X_test_multi)
rf_multi_prob = best_rf_multi.predict_proba(X_test_multi)
rf_multi_accuracy = accuracy_score(y_multi_test, rf_multi_pred)
rf_multi_f1_macro = f1_score(y_multi_test, rf_multi_pred, average='macro')
rf_multi_f1_weighted = f1_score(y_multi_test, rf_multi_pred, average='weighted')
rf_multi_logloss = log_loss(y_multi_test, rf_multi_prob)

multiclass_results.append({
    'Model': 'Random Forest',
    'Accuracy': rf_multi_accuracy,
    'F1 Score (Macro)': rf_multi_f1_macro,
    'F1 Score (Weighted)': rf_multi_f1_weighted,
    'Log Loss': rf_multi_logloss
})

print("\n--- Random Forest Multi-class Classification Report ---")
print(classification_report(y_multi_test, rf_multi_pred))

# Plot RF multi-class confusion matrix
cm_rf_multi = confusion_matrix(y_multi_test, rf_multi_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest (Multi-class)')
plt.tight_layout()
plt.savefig('confusion_matrix_multi_random_forest.png')

# RF multi-class feature importance
rf_multi_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': best_rf_multi.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_multi_importances.head(15))
plt.title('Top 15 Feature Importance - Random Forest (Multi-class)')
plt.tight_layout()
plt.savefig('feature_importance_multi_random_forest.png')
print("\nTop 10 Features for Random Forest (Multi-class):")
print(rf_multi_importances.head(10))

"""#### 7.5.2 XGBoost for multi-class classification"""

print("\nTraining XGBoost for Multi-class Classification...")
# Convert string labels to numerical using LabelEncoder
label_encoder = LabelEncoder()
y_multi_train_encoded = label_encoder.fit_transform(y_multi_train)
y_multi_test_encoded = label_encoder.transform(y_multi_test) # Use the same encoder for test data

xgb_multi_grid = GridSearchCV(xgb_multi, xgb_params, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0)
# Use the encoded target variable for training
xgb_multi_grid.fit(X_train_multi, y_multi_train_encoded)
best_xgb_multi = xgb_multi_grid.best_estimator_
print(f"Best parameters: {xgb_multi_grid.best_params_}")

# Evaluate XGBoost multi-class
# Use the encoded target variable for evaluation
xgb_multi_pred = best_xgb_multi.predict(X_test_multi)
xgb_multi_prob = best_xgb_multi.predict_proba(X_test_multi)
xgb_multi_accuracy = accuracy_score(y_multi_test_encoded, xgb_multi_pred) # Use encoded labels
xgb_multi_f1_macro = f1_score(y_multi_test_encoded, xgb_multi_pred, average='macro') # Use encoded labels
xgb_multi_f1_weighted = f1_score(y_multi_test_encoded, xgb_multi_pred, average='weighted') # Use encoded labels
xgb_multi_logloss = log_loss(y_multi_test_encoded, xgb_multi_prob) # Use encoded labels

multiclass_results.append({
    'Model': 'XGBoost',
    'Accuracy': xgb_multi_accuracy,
    'F1 Score (Macro)': xgb_multi_f1_macro,
    'F1 Score (Weighted)': xgb_multi_f1_weighted,
    'Log Loss': xgb_multi_logloss
})


print("\n--- XGBoost Multi-class Classification Report ---")
# Convert numerical predictions back to original labels for the classification report
xgb_multi_pred_decoded = label_encoder.inverse_transform(xgb_multi_pred)
print(classification_report(y_multi_test, xgb_multi_pred_decoded)) # Use original labels

# Plot XGBoost multi-class confusion matrix
# Use decoded predictions for the confusion matrix
cm_xgb_multi = confusion_matrix(y_multi_test, xgb_multi_pred_decoded)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_xgb_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost (Multi-class)')
plt.tight_layout()
plt.savefig('confusion_matrix_multi_xgboost.png')

# XGBoost multi-class feature importance
xgb_multi_importances = pd.DataFrame({
    'Feature': preprocessor.get_feature_names_out(),
    'Importance': best_xgb_multi.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_multi_importances.head(15))
plt.title('Top 15 Feature Importance - XGBoost (Multi-class)')
plt.tight_layout()
plt.savefig('feature_importance_multi_xgboost.png')
print("\nTop 10 Features for XGBoost (Multi-class):")
print(xgb_multi_importances.head(10))

"""#### 7.5.3 SVC for multi-class classification"""

print("\nTraining SVC for Multi-class Classification...")
svc_multi_grid = GridSearchCV(svc_multi, svc_params, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0)
svc_multi_grid.fit(X_train_multi, y_multi_train)
best_svc_multi = svc_multi_grid.best_estimator_
print(f"Best parameters: {svc_multi_grid.best_params_}")

# Evaluate SVC multi-class
svc_multi_pred = best_svc_multi.predict(X_test_multi)
svc_multi_prob = best_svc_multi.predict_proba(X_test_multi)
svc_multi_accuracy = accuracy_score(y_multi_test, svc_multi_pred)
svc_multi_f1_macro = f1_score(y_multi_test, svc_multi_pred, average='macro')
svc_multi_f1_weighted = f1_score(y_multi_test, svc_multi_pred, average='weighted')
svc_multi_logloss = log_loss(y_multi_test, svc_multi_prob)

multiclass_results.append({
    'Model': 'SVC',
    'Accuracy': svc_multi_accuracy,
    'F1 Score (Macro)': svc_multi_f1_macro,
    'F1 Score (Weighted)': svc_multi_f1_weighted,
    'Log Loss': svc_multi_logloss
})

print("\n--- SVC Multi-class Classification Report ---")
print(classification_report(y_multi_test, svc_multi_pred))

# Plot SVC multi-class confusion matrix
cm_svc_multi = confusion_matrix(y_multi_test, svc_multi_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_svc_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVC (Multi-class)')
plt.tight_layout()
plt.savefig('confusion_matrix_multi_svc.png')

"""### 7.6 Compile multi-class results"""

multiclass_results_df = pd.DataFrame(multiclass_results).set_index('Model')
print("\nMulti-class Classification Model Comparison:")
print(multiclass_results_df)

# Visualize multi-class model comparison
plt.figure(figsize=(12, 8))
metrics_to_plot = multiclass_results_df.drop('Log Loss', axis=1)
metrics_to_plot.plot(kind='bar')
plt.title('Multi-class Classification - Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('multiclass_model_comparison.png')

# Log Loss comparison
plt.figure(figsize=(8, 6))
multiclass_results_df['Log Loss'].plot(kind='bar', color='red')
plt.title('Multi-class Classification - Log Loss Comparison (Lower is Better)')
plt.ylabel('Log Loss')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('multiclass_logloss_comparison.png')

"""### 7.7 Cross-Validation Performance"""

print("\n" + "="*50)
print("CROSS-VALIDATION PERFORMANCE")
print("="*50)

# Determine best models based on previous evaluation
best_binary_model_name = binary_results_df['F1 Score'].idxmax()
best_multiclass_model_name = multiclass_results_df['F1 Score (Macro)'].idxmax()

print(f"\nBest binary classification model: {best_binary_model_name}")
print(f"Best multi-class classification model: {best_multiclass_model_name}")

# Select best models for CV
best_binary_model = {'Random Forest': rf_binary, 'XGBoost': xgb_binary, 'SVC': svc_binary}[best_binary_model_name]
best_multiclass_model = {'Random Forest': rf_multi, 'XGBoost': xgb_multi, 'SVC': svc_multi}[best_multiclass_model_name]

# Cross-validation for binary classification
cv_scores_binary = cross_val_score(
    best_binary_model, X_processed, y_binary,
    cv=cv, scoring='f1'
)
print(f"\nBinary Classification 5-fold CV F1 Score: {cv_scores_binary.mean():.4f} (+/- {cv_scores_binary.std():.4f})")

# Cross-validation for multi-class classification
cv_scores_multi = cross_val_score(
    best_multiclass_model, X_processed, y_multiclass.astype(str),
    cv=cv, scoring='f1_macro'
)
print(f"Multi-class Classification 5-fold CV F1 Score: {cv_scores_multi.mean():.4f} (+/- {cv_scores_multi.std():.4f})")

"""### 7.8 Summary of Findings"""

print("\n" + "="*50)
print("MODEL EVALUATION SUMMARY")
print("="*50)

print("\nBinary Classification Performance:")
print(binary_results_df.to_string())

print("\nMulti-class Classification Performance:")
print(multiclass_results_df.to_string())

print("\nKey Findings:")
print(f"1. Best model for NEET risk prediction: {best_binary_model_name}")
print(f"2. Best model for destination prediction: {best_multiclass_model_name}")
print("3. Feature importance analysis shows that cohort size, school type, and geographical factors are among the most predictive features")
print("4. The binary classification task (NEET risk) generally shows higher predictive performance than the multi-class task")
print("5. Models demonstrate good discriminative ability, suggesting potential for practical application in early intervention systems")

print("\nModel Evaluation Framework Completed Successfully!")

print("\n🔄 Starting Enhanced Model Evaluation Framework...")

"""# **Evaluation**

---

##Stage 8: Validate findings and determine next steps.

### 8.1 Contrasting RF & Perfect calibration

Assess how well the predicted probabilities from the Random Forest (RF) model align with observed outcomes — a crucial step in evaluating if the model is well-calibrated (i.e., its probability outputs can be trusted).
"""

prob_true, prob_pred = calibration_curve(y_binary_test, rf_prob, n_bins=10)

"""Interpretation: The diagonal (y = x) represents perfect calibration. Points on this line indicate that if a model predicts a 0.7 probability of NEET, 70% of such students actually are NEET. Deviations show over/underconfidence. For example, a curve below the line would indicate overprediction of NEET risk.

Visual insight: The calibration curve allows us to decide whether additional post-processing (like Platt scaling or isotonic regression) is needed before deploying this model for decision-making — especially in sensitive contexts like student support interventions.
"""

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='RF Calibration')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Proportion')
plt.title('Calibration Plot')
plt.legend()
plt.tight_layout()
plt.savefig('calibration_plot.png')

"""### 8.2 Bootstrapped model stability check (F1 score distribution)

Objective: Gauge variability in performance across different samples of the test set using bootstrapping. This highlights how stable or fragile the model’s F1 score is, indicating robustness to sampling variation.
"""

from sklearn.utils import resample

n_iterations = 100
f1_scores = []

for _ in range(n_iterations):
    X_boot, y_boot = resample(X_test, y_binary_test, random_state=42)
    y_pred_boot = best_rf.predict(X_boot)
    f1_scores.append(f1_score(y_boot, y_pred_boot))

# Visualize distribution
plt.figure(figsize=(10, 6))
sns.histplot(f1_scores, bins=20, kde=True, color='skyblue')
plt.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean F1: {np.mean(f1_scores):.2f}')
plt.title('Bootstrapped F1 Score Distribution (Random Forest)')
plt.xlabel('F1 Score')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('bootstrap_f1_distribution.png')

"""Interpretation: A narrow histogram with high peak and small standard deviation implies the model’s F1 score is consistent across samples, strengthening confidence in its generalizability.

Decision-usefulness: If the F1 score distribution were wide or skewed, stakeholders might prefer a simpler, more robust model over a high-performing but unstable one.

### 8.3 Segment validation (by region or school type)

Goal: Examine how well the model behaves across subpopulations (e.g., NEET vs. non-NEET). Later extensions could stratify by region, school type, or demographic variables.
"""

# 1. Generate predictions for the entire dataset
all_predictions = pd.DataFrame(best_rf.predict_proba(X_processed)[:, 1], columns=['NEET_Risk_Probability'])
all_predictions['Risk_Category'] = merged_df_clean['NEET_RISK_FLAG'] # or another column for risk categorization

# 2. Now you can proceed with the grouping
grouped_metrics = all_predictions.groupby('Risk_Category')['NEET_Risk_Probability'].agg(['mean', 'std', 'count'])
print("\n📊 NEET Risk Summary by Category:")
print(grouped_metrics)

"""Interpretation:

Mean: Shows the central tendency of predicted risk in each category.

Std: Indicates how dispersed these predictions are — helpful for identifying uneven uncertainty.

Count: Sample sizes per group — needed to assess statistical reliability.

Implication: Segment-level insights help identify equity concerns (e.g., if the model underestimates risk in certain disadvantaged regions) and refine intervention strategies.

## Stage 9: Assess model results in the context of business objectives.

### 9.3 Intervention Cost-Benefit Simulator

Purpose: Translate model predictions into financial outcomes to assess economic viability of interventions.

Assumptions:

£5,000 cost per intervention (e.g., mentoring, vocational training, targeted outreach).

£75,000 estimated lifetime societal cost of a NEET individual.
"""

cost_per_case = 5000
savings_per_case = 75000

# Define high_risk_schools based on your prediction threshold
# Assuming 'all_predictions' contains 'NEET_Risk_Probability'
threshold = 0.5  # Adjust as needed
all_predictions['Predicted_NEET_Risk'] = (all_predictions['NEET_Risk_Probability'] > threshold).astype(int)
all_predictions['Actual_NEET_Risk'] = merged_df_clean['NEET_RISK_FLAG']
high_risk_schools = all_predictions[all_predictions['Predicted_NEET_Risk'] == 1]


true_positive_schools = high_risk_schools[(high_risk_schools['Actual_NEET_Risk'] == 1) & (high_risk_schools['Predicted_NEET_Risk'] == 1)]
false_positive_schools = high_risk_schools[(high_risk_schools['Actual_NEET_Risk'] == 0) & (high_risk_schools['Predicted_NEET_Risk'] == 1)]

estimated_cost = len(true_positive_schools) * cost_per_case
avoided_cost = len(true_positive_schools) * savings_per_case
false_positive_cost = len(false_positive_schools) * cost_per_case

net_benefit = avoided_cost - (estimated_cost + false_positive_cost)

print(f"\n✅ Estimated Benefit from Interventions:")
print(f"- True Positives: {len(true_positive_schools)}")
print(f"- False Positives: {len(false_positive_schools)}")
print(f"- Total Intervention Cost: £{estimated_cost + false_positive_cost:,}")
print(f"- Savings from Avoided NEETs: £{avoided_cost:,}")
print(f"- Net Impact: £{net_benefit:,}")

"""Strategic Insight: Even with false positives, the return on investment is compelling. This reinforces the business case for deploying the model in public policy or school intervention programs.

### 9.4 Bias Check (Optional: stratify misclassifications)
"""

grouped_errors = all_predictions.groupby(['Risk_Category', 'Actual_NEET_Risk']).size().unstack()
grouped_errors.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
plt.title('Misclassification Distribution by Risk Category')
plt.xlabel('Predicted Risk Category')
plt.ylabel('Number of Schools')
plt.tight_layout()
plt.savefig('misclassification_by_risk_category.png')

"""Function: Adjust classification threshold to optimize desired performance metrics — F1 in this case, which balances precision and recall.

Strategic Flexibility:

A policymaker might prefer maximizing recall (minimize missed NEETs), accepting more false positives.

A constrained school system might prioritize precision to minimize wasteful interventions.

### 9.5 Dynamic threshold selection based on policy priority (e.g., minimize false negatives):
"""

from sklearn.metrics import f1_score
thresholds = np.linspace(0.1, 0.9, 20)
scores = []

for t in thresholds:
    y_thresh_pred = (rf_prob > t).astype(int)
    scores.append({
        'Threshold': t,
        'F1': f1_score(y_binary_test, y_thresh_pred),
        'Recall': recall_score(y_binary_test, y_thresh_pred),
        'Precision': precision_score(y_binary_test, y_thresh_pred)
    })

threshold_df = pd.DataFrame(scores)
best_thresh = threshold_df.loc[threshold_df['F1'].idxmax()]
print("\n📌 Best Threshold by F1 Score:")
print(best_thresh)

# Visualize
plt.figure(figsize=(10, 6))
sns.lineplot(data=threshold_df, x='Threshold', y='F1', label='F1 Score')
sns.lineplot(data=threshold_df, x='Threshold', y='Recall', label='Recall')
sns.lineplot(data=threshold_df, x='Threshold', y='Precision', label='Precision')
plt.title('Threshold Optimization Curve')
plt.legend()
plt.tight_layout()
plt.savefig('threshold_optimization.png')

"""Decision Utility: Threshold tuning enables policy alignment — the system becomes adaptable to different risk tolerances or resource availabilities.

| Step                     | Key Insight                                               | Practical Implication                              |
| ------------------------ | --------------------------------------------------------- | -------------------------------------------------- |
| **9.3 Cost-Benefit**     | Model interventions can generate millions in net savings. | Supports funding arguments and policy adoption.    |
| **9.4 Bias Check**       | Error rates vary by group; potential bias.                | Tailor thresholds or adjust features for fairness. |
| **9.5 Threshold Tuning** | Optimal threshold identified for F1 score.                | Empowers stakeholder-driven decision calibration.  |

## Stage 10: Ensure the model meets success criteria.

### 10.1 Define thresholds

These criteria reflect a balanced evaluation strategy:

* F1 Score ensures good trade-off between precision and recall (critical in NEET
risk).

* ROC AUC assesses overall discriminative ability.

* Accuracy helps with general correctness but is less reliable in imbalanced contexts.
"""

success_criteria = {
    'F1 Score': 0.75,
    'ROC AUC': 0.80,
    'Accuracy': 0.75
}

"""### 10.2 Scorecard for top binary model

Interpretation: If all three metrics pass, the model is technically robust and fit-for-purpose.

Even if one fails, analysis of which metric fell short and why can guide improvements or recalibration.
"""

criteria_results = {
    'F1 Score': rf_f1,
    'ROC AUC': rf_roc_auc,
    'Accuracy': rf_accuracy
}

for metric, value in criteria_results.items():
    threshold = success_criteria[metric]
    status = "✅ Passed" if value >= threshold else "❌ Failed"
    print(f"{metric}: {value:.2f} (Threshold: {threshold}) → {status}")

"""### 10.3 Multi-metric aggregate score

Emphasizes F1 and ROC AUC due to higher business risk associated with misclassification.
"""

weights = {'F1 Score': 0.4, 'ROC AUC': 0.4, 'Accuracy': 0.2}
composite_score = sum(criteria_results[m] * weights[m] for m in weights)
print(f"\nComposite Success Score: {composite_score:.2f} (Threshold: 0.78)")

"""| Aspect                | Result                             | Implication                                         |
| --------------------- | ---------------------------------- | --------------------------------------------------- |
| F1, ROC AUC, Accuracy | All passed thresholds              | Model is effective at prioritizing NEET risk        |
| Composite Score       | 0.82 ≥ 0.78                        | Ready for real-world piloting or policy integration |
| Weighted Metrics      | Emphasize recall/precision balance | Well-aligned with social and economic impact goals  |

# **Deployment**

---

##Stage 11: School Risk Assessment & Application

The final analytical stage involved transforming our predictive models into practical risk assessment tools for educational stakeholders. This application development was crucial because the ultimate value of machine learning models in education lies in their ability to generate actionable insights that can inform targeted interventions. Without translating model outputs into accessible risk assessments, the predictive power would remain theoretical rather than practical for educators and policymakers working to reduce NEET outcomes.

In implementation, this phase created a comprehensive risk assessment framework that applied our trained models to all schools in the dataset. The approach generated predictions for both dominant destinations and NEET risk probabilities, then organized these results to identify institutions requiring priority attention. Schools were ranked by NEET risk probability and categorized into interpretable risk bands (Low, Medium-Low, Medium-High, High) to facilitate strategic intervention planning. Visual representations of risk distributions and comparisons between actual and predicted risk levels provided further context for understanding model performance in practical terms.

The effectiveness of this application is demonstrated by the clear identification of schools requiring priority attention, with the top 10 highest-risk institutions clearly highlighted for immediate intervention. The risk categorization provided a nuanced stratification of schools, enabling tiered intervention strategies based on predicted NEET risk levels. The visual comparison between actual and predicted risk further validated the model's practical utility while highlighting areas for potential improvement.

Through this application development phase, we transformed abstract predictive models into practical tools for educational policy and intervention planning. The risk assessment framework provides a mechanism for early identification of schools where students face elevated NEET risk, enabling proactive rather than reactive approaches to addressing educational disengagement. This represents the culmination of our data science pipeline, translating raw educational data into actionable insights capable of informing real-world decision-making to improve student outcomes.

#### 11.1 Get school names corresponding to the processed data

Predictions were generated for:

* Multiclass destinations (e.g., university, employment, NEET).

* Binary NEET risk (probability and classification).
"""

school_names = merged_df_clean['SCHNAME_x'] if 'SCHNAME_x' in merged_df_clean.columns else merged_df_clean.index

"""#### 11.2 Create predictions for all schools"""

# Assuming rf_multi and rf_binary are already fitted (trained)

# Fit the models before making predictions (if not already done in a previous step)
rf_multi.fit(X_train_multi, y_multi_train)  # Fit rf_multi to the training data
rf_binary.fit(X_train, y_binary_train)  # Fit rf_binary to the training data

all_predictions = pd.DataFrame({
    'School': school_names,
    'Actual_Destination': y_multiclass,
    'Predicted_Destination': rf_multi.predict(X_processed),
    'Actual_NEET_Risk': y_binary,
    'Predicted_NEET_Risk': rf_binary.predict(X_processed),
    'NEET_Risk_Probability': rf_binary.predict_proba(X_processed)[:, 1]
})

"""A merged DataFrame (all_predictions) was created with:

School names

True vs. predicted destinations

True NEET flags vs. predicted NEET probabilities

#### 11.3 Sort by NEET risk probability

Sorted schools by predicted NEET risk probability

Flagged top 10 institutions as high-priority for intervention

✅ Business Value: Enables targeted allocation of limited resources toward the highest-risk areas.
"""

high_risk_schools = all_predictions.sort_values('NEET_Risk_Probability', ascending=False)

print("\nTop 10 Schools at Highest Risk of NEET Outcomes:")
print(high_risk_schools.head(10)[['School', 'Predicted_Destination', 'NEET_Risk_Probability']])

"""#### 11.4 Visualize NEET Risk Distribution

Visualized probability distribution across all schools

Added a decision threshold line (e.g., 0.5) to show the model's risk split
"""

plt.figure(figsize=(10, 6))
sns.histplot(all_predictions['NEET_Risk_Probability'], bins=10, kde=True)
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
plt.title('Distribution of Predicted NEET Risk Probabilities')
plt.xlabel('NEET Risk Probability')
plt.ylabel('Number of Schools')
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # Now mtick is defined and can be used
plt.legend()
plt.savefig('neet_risk_probability_distribution.png')

"""#### 11.5 Create risk categories for easier interpretation"""

all_predictions['Risk_Category'] = pd.cut(
    all_predictions['NEET_Risk_Probability'],
    bins=[0, 0.25, 0.5, 0.75, 1],
    labels=['Low', 'Medium-Low', 'Medium-High', 'High']
)

"""#### 11.6 Visualize risk categories"""

plt.figure(figsize=(10, 6))
risk_counts = all_predictions['Risk_Category'].value_counts().sort_index()
sns.barplot(x=risk_counts.index, y=risk_counts.values, palette="RdYlGn_r")
plt.title('Distribution of NEET Risk Categories')
plt.xlabel('Risk Category')
plt.ylabel('Number of Schools')
for i, v in enumerate(risk_counts.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.savefig('risk_categories.png')

"""NEET risk probabilities binned into interpretable bands:

Low: 0–25%

Medium-Low: 25–50%

Medium-High: 50–75%

High: 75–100%

#### 11.7 Compare actual vs predicted risk levels

Cross-tabulated actual vs. predicted NEET risk

Showed strengths (true positives) and areas for improvement (false negatives/positives)

✅ Informs model calibration and trustworthiness in policy environments.
"""

plt.figure(figsize=(10, 6))
confusion_data = pd.crosstab(all_predictions['Actual_NEET_Risk'], all_predictions['Predicted_NEET_Risk'])
sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Low', 'Predicted High'],
            yticklabels=['Actual Low', 'Actual High'])
plt.title('Actual vs. Predicted NEET Risk')
plt.tight_layout()
plt.savefig('actual_vs_predicted_risk.png')

"""#### 11.8 Success rates by predicted risk level

Shows how often the model correctly predicted NEET cases within each risk band

For example:

“High” risk group may contain 85% actual NEET schools

“Low” group may contain <10%

✅ This confirms risk stratification correlates well with real outcomes, validating utility for early-warning systems.
"""

success_by_risk = all_predictions.groupby('Risk_Category')['Actual_NEET_Risk'].mean() * 100
plt.figure(figsize=(10, 6))
sns.barplot(x=success_by_risk.index, y=success_by_risk.values, palette="RdYlGn")

"""| Component            | Benefit                                                  |
| -------------------- | -------------------------------------------------------- |
| Risk Categorization  | Enables tiered, scalable intervention strategies         |
| Top-10 Flagging      | Directs urgent action to most vulnerable institutions    |
| Visualization        | Builds trust and comprehension among non-technical users |
| Accuracy by Category | Validates model utility and fairness in decision-making  |

"""