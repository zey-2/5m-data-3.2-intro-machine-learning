# Assignment

## Instructions

Complete the following exercises using Python.

1. Linear Regression Exercise:
   Using the California Housing dataset from scikit-learn, create a linear regression model to predict house prices.
   Evaluate the performance of Linear Regression on test set.

   ```python
   from sklearn.datasets import fetch_california_housing
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error, r2_score

   # Load dataset
   housing = fetch_california_housing()

   # Split features and target
   X = housing.data
   y = housing.target

   # Split into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Create and train the model
   model = LinearRegression()
   model.fit(X_train, y_train)

   # Make predictions
   y_pred = model.predict(X_test)

   # Evaluate the model
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)

   print(f"Mean Squared Error: {mse:.2f}")
   print(f"R2 Score: {r2:.2f}")
   ```

2. Classification Exercise:
   Using the breast cancer dataset from scikit-learn, build classification models to predict malignant vs benign tumors.
   Compare Logistic Regression and KNN performance on test set.

   ```python
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.metrics import accuracy_score, classification_report

   # Load dataset
   cancer = load_breast_cancer()

   # Split features and target
   X = cancer.data
   y = cancer.target

   # Split into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Train and evaluate Logistic Regression
   log_reg = LogisticRegression(random_state=42)
   log_reg.fit(X_train, y_train)
   log_reg_pred = log_reg.predict(X_test)

   print("Logistic Regression Results:")
   print(f"Accuracy: {accuracy_score(y_test, log_reg_pred):.2f}")
   print("\nClassification Report:")
   print(classification_report(y_test, log_reg_pred))

   # Train and evaluate KNN
   knn = KNeighborsClassifier(n_neighbors=5)
   knn.fit(X_train, y_train)
   knn_pred = knn.predict(X_test)

   print("\nKNN Results:")
   print(f"Accuracy: {accuracy_score(y_test, knn_pred):.2f}")
   print("\nClassification Report:")
   print(classification_report(y_test, knn_pred))
   ```

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
