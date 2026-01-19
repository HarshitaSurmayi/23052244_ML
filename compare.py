# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Load dataset
df = pd.read_csv("student_performance.csv")

# -------------------------------
# REGRESSION SETUP
# -------------------------------
X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression (single feature)
lr_single = LinearRegression()
lr_single.fit(X_train[["study_hours"]], y_train)
pred_lr_single = lr_single.predict(X_test[["study_hours"]])
rmse_lr_single = np.sqrt(mean_squared_error(y_test, pred_lr_single))

# 2. Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)
pred_mlr = mlr.predict(X_test)
rmse_mlr = np.sqrt(mean_squared_error(y_test, pred_mlr))

# 3. KNN Regression
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_scaled, y_train)
pred_knn_reg = knn_reg.predict(X_test_scaled)
rmse_knn = np.sqrt(mean_squared_error(y_test, pred_knn_reg))

# 4. Decision Tree Regression
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train, y_train)
pred_dt_reg = dt_reg.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, pred_dt_reg))

# -------------------------------
# CLASSIFICATION SETUP
# -------------------------------
df["pass_fail"] = (df["final_score"] >= 60).astype(int)

Xc = df[["study_hours", "attendance", "previous_score"]]
yc = df["pass_fail"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.3, random_state=42
)

Xc_train_scaled = scaler.fit_transform(Xc_train)
Xc_test_scaled = scaler.transform(Xc_test)

# 5. KNN Classification
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(Xc_train_scaled, yc_train)
pred_knn_clf = knn_clf.predict(Xc_test_scaled)
acc_knn = accuracy_score(yc_test, pred_knn_clf)

# 6. Naive Bayes
nb = GaussianNB()
nb.fit(Xc_train, yc_train)
pred_nb = nb.predict(Xc_test)
acc_nb = accuracy_score(yc_test, pred_nb)

# 7. Decision Tree Classification
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(Xc_train, yc_train)
pred_dt_clf = dt_clf.predict(Xc_test)
acc_dt = accuracy_score(yc_test, pred_dt_clf)

# -------------------------------
# RESULTS
# -------------------------------
print("REGRESSION (RMSE)")
print("Linear Regression (single):", rmse_lr_single)
print("Multiple Linear Regression:", rmse_mlr)
print("KNN Regression:", rmse_knn)
print("Decision Tree Regression:", rmse_dt)

print("\nCLASSIFICATION (Accuracy)")
print("KNN Classifier:", acc_knn)
print("Naive Bayes:", acc_nb)
print("Decision Tree Classifier:", acc_dt)
