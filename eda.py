import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# visual settings
plt.rcParams['figure.figsize'] = (10, 6)

# 2. LOAD DATA

#df = pd.read_csv("D:\Claudia\PythonProjects\Gym_exercise_dataset_analysis\data\gym_members_exercise_tracking.csv")
df = pd.read_csv("data/gym_members_exercise_tracking.csv")

# 3. BASIC INFO

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())

print("\nFirst rows:")
print(df.head())

# 4. MISSING VALUES

print("\nMissing values:")
print(df.isnull().sum())

# 5. DUPLICATES

print("\nDuplicate rows:", df.duplicated().sum())


# 6. DATA TYPES SEPARATION

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['str']).columns

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# 7. CATEGORICAL ANALYSIS

for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())

    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())


# CATEGORICAL CLEANLINESS CHECK

print("\nChecking categorical inconsistencies:")

for col in categorical_cols:
    print(f"\n--- {col} ---")

    # unique values
    unique_vals = df[col].unique()
    print("Unique values:", unique_vals)

    # spaces verification
    has_spaces = df[col].str.contains(" ").sum()
    print("Values containing spaces:", has_spaces)

    # lowercase/uppercase verification
    print("Lowercase version:")
    print(df[col].str.lower().value_counts())

    # verification of different lengths (posibil typo)
    print("Value lengths:")
    print(df[col].str.len().value_counts())


# 8. DESCRIPTIVE STATISTICS

print("\nDescriptive statistics:")
print(df.describe())


# 9. DISTRIBUTIONS (HISTOGRAMS)

df[numerical_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle("Distributions of Numerical Features")
plt.show()


# 10. BOXPLOTS (OUTLIERS)

for col in numerical_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()


# 11. CORRELATION MATRIX

corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# 12. RELATIONSHIPS (SCATTER)

# Frequency vs other variables
for col in numerical_cols:
    if col != 'Workout_Frequency (days/week)':
        plt.figure()
        sns.scatterplot(x=df[col], y=df['Workout_Frequency (days/week)'])
        plt.title(f"{col} vs Workout_Frequency (days/week)")
        plt.show()


# 13. CATEGORICAL vs NUMERICAL

for col in categorical_cols:
    plt.figure()
    sns.boxplot(x=df[col], y=df['Workout_Frequency (days/week)'])
    plt.title(f"{col} vs Workout_Frequency (days/week)")
    plt.show()


# 14. CLASS DISTRIBUTION (if needed)

for col in categorical_cols:
    plt.figure()
    sns.countplot(x=df[col])
    plt.title(f"Countplot for {col}")
    plt.show()


# 15. OUTLIER DETECTION (IQR)

print("\nOutliers (IQR method):")

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]

    print(f"{col}: {len(outliers)} outliers")


# 16. LOGICAL CONSISTENCY CHECKS

print("\nLogical checks:")

# Avg BPM <= Max BPM
invalid_bpm = df[df['Avg_BPM'] > df['Max_BPM']]
print("Rows where Avg_BPM > Max_BPM:", len(invalid_bpm))

# Resting BPM < Avg BPM
invalid_rest = df[df['Resting_BPM'] > df['Avg_BPM']]
print("Rows where Resting_BPM > Avg_BPM:", len(invalid_rest))

# BMI recalculation check
df['Calculated_BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)

difference = np.abs(df['BMI'] - df['Calculated_BMI'])
print("Average BMI difference:", difference.mean())


# TARGET ANALYSIS

print("\n=== TARGET ANALYSIS ===")

target = 'Workout_Frequency (days/week)'

print(f"Selected target: {target}")

# problem type
if df[target].dtype in ['int64', 'float64']:
    print("Problem type: Regression")
else:
    print("Problem type: Classification")

# target distribution
plt.figure()
sns.histplot(df[target], bins=20, kde=True)
plt.title(f"Distribution of {target}")
plt.show()

# correlations with target
print("\nCorrelation with target:")
print(df.corr(numeric_only=True)[target].sort_values(ascending=False))
