import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('Housing.csv')

# Handle NaN values if present
df = df.dropna()  

# Perform one-hot encoding
df = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

# Display correlation matrix
correlation_matrix = df.corr()

# Check for NaN values in the correlation matrix
if correlation_matrix.isnull().values.any():
    print("Correlation matrix contains NaN values. Handle NaN values before creating the heatmap.")
else:
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation with Price")
    plt.show()

# Adjust feature selection based on column names after one-hot encoding
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'airconditioning_yes', 'parking', 'prefarea_yes']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=40)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predicted_prices = model.predict(X_test)
accuracy = r2_score(y_test, predicted_prices)
print("R-squared:", accuracy)