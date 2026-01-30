import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# Parse all the sequences
sequences = [
    "6: 7, 7: 9, 8: 14, 9: 17, 10: 21, 11: 26, 12: 32, 13: 37, 14: 42, 15: 48, 16: 55, 17: 61, 18: 68, 19: 74, 20: 82, 21: 89, 22: 97, 23: 104, 24: 113, 25: 122, 26: 133, 27: 142, 28: 153",
    "6: 7, 7: 10, 8: 14, 9: 17, 10: 20, 11: 25, 12: 31, 13: 37, 14: 44, 15: 51, 16: 58, 17: 66, 18: 72, 19: 80, 20: 88, 21: 96, 22: 103, 23: 112, 24: 121, 25: 131, 26: 141, 27: 150, 28: 159",
    "7: 10, 8: 13, 9: 17, 10: 20, 11: 25, 12: 33, 13: 36, 14: 42, 15: 49, 16: 56, 17: 65, 18: 74, 19: 83, 20: 93, 21: 102, 22: 111, 23: 119, 24: 129, 25: 138, 26: 146, 27: 156, 28: 166",
    "8: 13, 9: 16, 10: 21, 11: 25, 12: 33, 13: 37, 14: 42, 15: 48, 16: 54, 17: 61, 18: 70, 19: 78, 20: 89, 21: 100, 22: 111, 23: 124, 24: 136, 25: 147, 26: 157, 27: 167, 28: 178",
    "9: 17, 10: 21, 11: 26, 12: 31, 13: 36, 14: 42, 15: 50, 16: 59, 17: 62, 18: 69, 19: 76, 20: 86, 21: 96, 22: 105, 23: 117, 24: 130, 25: 143, 26: 157, 27: 172, 28: 187",
    "10: 20, 11: 26, 12: 32, 13: 37, 14: 42, 15: 48, 16: 59, 17: 66, 18: 71, 19: 76, 20: 85, 21: 93, 22: 103, 23: 113, 24: 125, 25: 136, 26: 149, 27: 163, 28: 179",
    "11: 24, 12: 30, 13: 37, 14: 44, 15: 49, 16: 54, 17: 62, 18: 71, 19: 83, 20: 95, 21: 97, 22: 103, 23: 111, 24: 121, 25: 133, 26: 145, 27: 158, 28: 169",
    "12: 29, 13: 35, 14: 42, 15: 51, 16: 56, 17: 61, 18: 69, 19: 76, 20: 95, 21: 102, 22: 109, 23: 116, 24: 125, 25: 130, 26: 141, 27: 153, 28: 167",
    "13: 34, 14: 40, 15: 48, 16: 58, 17: 65, 18: 70, 19: 76, 20: 85, 21: 97, 22: 109, 23: 123, 24: 137, 25: 140, 26: 147, 27: 153, 28: 164",
    "14: 39, 15: 46, 16: 55, 17: 66, 18: 74, 19: 78, 20: 86, 21: 93, 22: 103, 23: 116, 24: 137, 25: 147, 26: 155, 27: 161, 28: 175",
    "15: 45, 16: 51, 17: 61, 18: 72, 19: 83, 20: 89, 21: 96, 22: 103, 23: 111, 24: 125, 25: 140, 26: 155, 27: 172, 28: 189",
    "16: 51, 17: 57, 18: 68, 19: 80, 20: 93, 21: 100, 22: 105, 23: 113, 24: 121, 25: 130, 26: 147, 27: 161, 28: 189",
    "17: 57, 18: 63, 19: 74, 20: 88, 21: 102, 22: 111, 23: 117, 24: 125, 25: 133, 26: 141, 27: 153, 28: 175",
    "18: 65, 19: 70, 20: 82, 21: 96, 22: 111, 23: 124, 24: 130, 25: 136, 26: 145, 27: 153, 28: 164",
    "19: 71, 20: 77, 21: 89, 22: 103, 23: 119, 24: 136, 25: 143, 26: 149, 27: 158, 28: 167",
    "20: 79, 21: 85, 22: 97, 23: 112, 24: 129, 25: 147, 26: 157, 27: 163, 28: 169",
    "21: 87, 22: 93, 23: 104, 24: 121, 25: 138, 26: 157, 27: 172, 28: 179",
    "22: 95, 23: 102, 24: 113, 25: 131, 26: 146, 27: 167, 28: 187",
]

# Parse data
data = []
for k, seq in enumerate(sequences, start=1):
    pairs = seq.split(", ")
    for pair in pairs:
        index, value = pair.split(": ")
        data.append([k, int(index), int(value)])

df = pd.DataFrame(data, columns=['k', 'index', 'value'])
print(f"Total data points: {len(df)}")
print(f"k range: {df['k'].min()} to {df['k'].max()}")
print(f"index range: {df['index'].min()} to {df['index'].max()}")
print(f"value range: {df['value'].min()} to {df['value'].max()}")
print("\nFirst few rows:")
print(df.head(10))
print("\nLast few rows:")
print(df.tail(10))

# Prepare features (k, index) and target (value)
X = df[['k', 'index']].values
y = df['value'].values

# Try different polynomial degrees
print("\n" + "="*60)
print("FITTING POLYNOMIAL MODELS")
print("="*60)

for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\n{'='*60}")
    print(f"DEGREE {degree} POLYNOMIAL")
    print(f"{'='*60}")
    print(f"R² score: {r2:.6f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Get feature names
    feature_names = poly.get_feature_names_out(['k', 'index'])
    
    print(f"\nFormula: value = ", end="")
    terms = []
    for i, (coef, name) in enumerate(zip(model.coef_, feature_names)):
        if i == 0:  # intercept (constant term from bias)
            if abs(coef) > 0.0001:
                terms.append(f"{coef:.4f}")
        else:
            if abs(coef) > 0.0001:
                # Clean up the feature name
                clean_name = name.replace(' ', '*')
                if coef >= 0:
                    terms.append(f"+ {coef:.4f}*{clean_name}")
                else:
                    terms.append(f"- {abs(coef):.4f}*{clean_name}")
    
    print(" ".join(terms))
    
    # Show some sample predictions vs actual
    if degree == 2:
        print("\nSample predictions (Degree 2):")
        print(f"{'k':<4} {'index':<6} {'Actual':<8} {'Predicted':<10} {'Error':<8}")
        print("-" * 50)
        sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, len(df)-1]
        for idx in sample_indices:
            if idx < len(df):
                k_val = df.iloc[idx]['k']
                index_val = df.iloc[idx]['index']
                actual = df.iloc[idx]['value']
                predicted = y_pred[idx]
                error = actual - predicted
                print(f"{k_val:<4} {index_val:<6} {actual:<8} {predicted:<10.2f} {error:<8.2f}")

# Check residuals for degree 2
print("\n" + "="*60)
print("RESIDUAL ANALYSIS (Degree 2)")
print("="*60)
poly2 = PolynomialFeatures(degree=2, include_bias=True)
X_poly2 = poly2.fit_transform(X)
model2 = LinearRegression()
model2.fit(X_poly2, y)
y_pred2 = model2.predict(X_poly2)
residuals = y - y_pred2

print(f"\nMean residual: {np.mean(residuals):.4f}")
print(f"Std of residuals: {np.std(residuals):.4f}")
print(f"Max absolute error: {np.max(np.abs(residuals)):.4f}")
print(f"95th percentile error: {np.percentile(np.abs(residuals), 95):.4f}")

# Show where the largest errors occur
print("\nLargest errors:")
df['predicted'] = y_pred2
df['error'] = residuals
df['abs_error'] = np.abs(residuals)
print(df.nlargest(10, 'abs_error')[['k', 'index', 'value', 'predicted', 'error']])