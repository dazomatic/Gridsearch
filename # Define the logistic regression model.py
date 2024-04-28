# Define the logistic regression model
lr = LogisticRegression()

# Define the hyperparameters grid to search
param_grid = {
    'C': [0.1, 1, 10],           # Regularization parameter
    'penalty': ['l1', 'l2'],     # Penalty (L1 or L2 regularization)
    'solver': ['liblinear']      # Algorithm to use in the optimization problem
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_lr = grid_search.best_estimator_

# Make predictions on the test set
predict = best_lr.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, predict)
print("Accuracy:", accuracy)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, predict)
print("Confusion Matrix:")
print(conf_matrix)