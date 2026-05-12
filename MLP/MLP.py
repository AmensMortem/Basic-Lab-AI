import matplotlib.pyplot as plt
import numpy as np


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, epochs, learning_rate):
        self.input_size, self.output_size, self.hidden_sizes = input_size, output_size, hidden_sizes
        self.epochs, self.learning_rate = epochs, learning_rate
        self.losses, self.activations, self.outputs = [], [], []
        self.weights, self.biases = [], []
        sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(sizes) - 1
        for i in range(self.num_layers):
            self.weights.append(np.random.randn(sizes[i + 1], sizes[i]) * np.sqrt(2 / sizes[i]))
            self.biases.append(np.random.randn(sizes[i + 1], 1))

    # Activation functions
    @staticmethod
    def tanh(Z):  # Hyperbolic tangent (tanh) activation function
        return np.tanh(Z)

    @staticmethod
    def conversion(x):
        return np.where(x >= 0.5, 1, 0)

    @staticmethod
    def gradient_tanh(Z):  # Gradient of the hyperbolic tangent (tanh) activation function
        return 1 - np.tanh(Z) ** 2

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):  # Forward pass through the network
        self.activations, self.outputs = [X], []
        for i in range(self.num_layers):
            output_ = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.outputs.append(output_)
            if i < self.num_layers - 1:
                output = self.tanh(output_)
            else:
                output = self.sigmoid(output_)
            self.activations.append(output)
        return self.activations[-1]

    def backward(self, X, y):
        num_trainy = X.shape[1]
        gradients = []
        delta_Z = self.activations[-1] - y
        for i in range(self.num_layers - 1, -1, -1):
            delta_W = (1 / num_trainy) * np.dot(delta_Z, self.activations[i].T)
            delta_B = (1 / num_trainy) * np.sum(delta_Z, axis=1, keepdims=True)
            gradients.append((delta_W, delta_B))
            if i > 0:
                delta_A = np.dot(self.weights[i].T, delta_Z)
                delta_Z = delta_A * self.gradient_tanh(self.outputs[i - 1])
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients[i][0]
            self.biases[i] -= self.learning_rate * gradients[i][1]

    @staticmethod
    def mse_loss(outputs, original):
        return np.mean(np.square(outputs - original))

    def train(self, X_train, y_train):
        for epoch in range(self.epochs):
            outputs = self.forward(X_train)
            loss = self.mse_loss(outputs, y_train)
            self.losses.append(loss)
            self.update_parameters(self.backward(X_train, y_train))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        pred = self.forward(X)
        return (pred >= 0.5).astype(int).flatten()

    @staticmethod
    def activation(_data):
        return np.where(_data >= 0, 1, 0)

    def visualizing_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.show()


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    n_informative=5,
    random_state=42
)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.T
X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T

mlp = MLP(
    input_size=10,
    hidden_sizes=[64, 32],
    output_size=1,
    epochs=2000,
    learning_rate=0.001
)

mlp.train(X_train, y_train)
predicts = mlp.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predicts):.4f}")
mlp.visualizing_loss()
