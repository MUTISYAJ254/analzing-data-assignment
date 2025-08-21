# ---------------------------------------------
# Assignment: Data Analysis & Visualization
# Objective: Load, analyze, and visualize a dataset
# Dataset: Iris dataset from sklearn
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------

try:
    # Load dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame  # Convert to pandas DataFrame

    print("✅ Dataset Loaded Successfully!\n")

    # Display first few rows
    print("🔹 First 5 Rows of Dataset:")
    print(df.head(), "\n")

    # Check data types & missing values
    print("🔹 Info about Dataset:")
    print(df.info(), "\n")

    print("🔹 Missing Values:")
    print(df.isnull().sum(), "\n")

    # Clean dataset (no missing values in Iris, but we'll handle if they exist)
    df = df.dropna()

except FileNotFoundError:
    print("❌ Error: Dataset file not found!")

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

print("🔹 Basic Statistics of Numerical Columns:")
print(df.describe(), "\n")

# Group by species & compute mean of numerical columns
grouped = df.groupby("target").mean()
print("🔹 Mean values by species:")
print(grouped, "\n")

# Map target numbers to species names
df["species"] = df["target"].map({i: name for i, name in enumerate(iris.target_names)})

print("🔹 Interesting Finding:")
print("Average petal length differs significantly among species!\n")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# Line Chart: Sepal length over index (not a time series, but shows trend)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, ci=None)
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Sepal length vs. Petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Scatter Plot: Sepal vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
