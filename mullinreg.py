import numpy as np

print("---- Multiple Linear Regression (Manual Formula) ----")


n = int(input("Enter number of data points: "))


m = int(input("Enter number of features (x1, x2, ..., xm): "))


X = []
y = []

print("\nEnter values for each data point:")
for i in range(n):
    row = []
    print(f"\n--- Data Point {i+1} ---")
    for j in range(m):
        val = float(input(f"Enter x{j+1}: "))
        row.append(val)
    X.append(row)

    target = float(input("Enter y value: "))
    y.append(target)


X = np.array(X)     
y = np.array(y)     


X_b = np.c_[np.ones((n, 1)), X]   

beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("\n===== MODEL OUTPUT =====")
print("Intercept (β0):", beta[0])
for i in range(1, len(beta)):
    print(f"Coefficient β{i} (for x{i}): {beta[i]}")


print("\n===== PREDICTION =====")
new_x = []

print("\nEnter feature values to predict y:")
for j in range(m):
    val = float(input(f"Enter x{j+1}: "))
    new_x.append(val)


new_x = np.array([1] + new_x)

predicted_y = new_x.dot(beta)
print("\nPredicted y =", predicted_y)
