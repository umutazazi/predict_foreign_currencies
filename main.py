import numpy as np
import pandas as pd

# Read the Excel document
file_path = 'Doviz_Satislari_20050101_20231205_Training_Set.xlsx'
data = pd.read_excel(file_path)

# Remove NaN values
data = data.dropna()

# Maximum row count
max_row_count = len(data)


#------------------------ [ PHASE 1 ] ------------------------#


# Use a loop to ask the user which row they want to see
while True:
    try:
        row_number = int(input("Enter the index of the row you want to see (enter a number between 0 and {} or -1 to exit): ".format(max_row_count - 1)))
        if row_number == -1:
            break  # End the loop when the user wants to log out
        if 0 <= row_number < max_row_count:
            specific_row = data.iloc[row_number]
            print("Selected row:")
            print(specific_row)
        else:
            print("Invalid index. Please enter a number between 0 and {}.".format(max_row_count - 1))
    except ValueError:
        print("Please enter a valid number.")



#----------------------- [ PHASE 2 ] ------------------------#


# Cost/Loss Function (Mean Squared Error - MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Derivative of Cost/Loss Function
def derivative_of_mse(y_true, y_pred, X):
    return -2 * np.dot(X.T, (y_true - y_pred)) / len(y_true)

# Simple Linear Regression Model
class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Number of samples and features
        n_samples, n_features = X.shape

        # Initial values for weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for _ in range(self.iterations):
            # Model prediction
            y_pred = self.predict(X)

            # Gradient calculation
            dw = derivative_of_mse(y, y_pred, X)
            db = -2 * np.sum(y - y_pred) / n_samples

            # Weight and bias update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Training and prediction using the model and optimization
def train_and_predict(X_train, y_train, X_test, learning_rate=0.01, iterations=1000):
    model = LinearRegressionModel(learning_rate, iterations)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def scale_features(X):
    for column in X.columns:
        X[column] = (X[column] - X[column].mean()) / X[column].std()
    return X

X = data.drop(['No', 'Tarih', 'TP DK SAR S YTL'], axis=1)
X_scaled = scale_features(X.copy())  # Scale the features
y = data['TP DK SAR S YTL'].values

# Split data into input and output
X = data.drop(['No', 'Tarih', 'TP DK SAR S YTL'], axis=1).values
y = data['TP DK SAR S YTL'].values

# 80% training, 20% testing
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the model and make predictions on the test data
predictions = train_and_predict(X_train, y_train, X_test, learning_rate=0.001, iterations=5000)

# Show the first three predictions
print("First 3 predictions:", predictions[:3])



#------------------------[ PHASE 3 ] -------------------------#


# Cost/Loss Function (MSE with L1 and L2 penalties)
def elastic_net_mse(y_true, y_pred, weights, l1_ratio, alpha):
    l1_penalty = l1_ratio * alpha * np.sum(np.abs(weights))
    l2_penalty = (1 - l1_ratio) * alpha * np.sum(weights ** 2)
    return np.mean((y_true - y_pred) ** 2) + l1_penalty + l2_penalty

# Derivative of Cost/Loss Function with L1 and L2 penalties
def derivative_of_elastic_net_mse(y_true, y_pred, X, weights, l1_ratio, alpha):
    mse_derivative = -2 * np.dot(X.T, (y_true - y_pred)) / len(y_true)
    l1_derivative = l1_ratio * alpha * np.sign(weights)
    l2_derivative = (1 - l1_ratio) * 2 * alpha * weights
    return mse_derivative + l1_derivative + l2_derivative

# Elastic Net Linear Regression Model
class ElasticNetLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, l1_ratio=0.5, alpha=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = self.predict(X)
            dw = derivative_of_elastic_net_mse(y, y_pred, X, self.weights, self.l1_ratio, self.alpha)
            db = -2 * np.sum(y - y_pred) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


#---------------------- [ TEST PHASE ] ----------------------#


# Create an instance of ElasticNetLinearRegression
test_model = ElasticNetLinearRegression(learning_rate=0.001, iterations=5000, l1_ratio=0.5, alpha=0.01)

# Train the model on the training data
test_model.fit(X_train, y_train)

# Ask the user for seven exchange rate values
print("Please enter seven exchange rate values:")
inputs = []
for i in range(1, 8):
    value = float(input(f"Exchange Rate {i}: "))
    inputs.append(value)

# Convert the user inputs to a numpy array
inputs_array = np.array([inputs])

# Make predictions using the model
prediction = test_model.predict(inputs_array)

# Print the prediction
print(f"Predicted output exchange rate: {prediction[0]}")
