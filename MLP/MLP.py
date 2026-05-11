import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, epochs, learning_rate):
        self.input_size, self.output_size, self.hidden_sizes = input_size, output_size, hidden_sizes
        self.epochs, self.learning_rate = epochs, learning_rate
        self.losses, self.activations, self.outputs = [], [], []

        self.weights, self.biases = [], []
        sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(sizes)
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i - 1]))
            self.biases.append(np.random.randn(sizes[i], 1))

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
        delta_Z = self.activations[-1]
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
        gradients_w, gradients_b = gradients
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    @staticmethod
    def mse_loss(outputs, original):
        return np.mean(np.square(outputs - original))

    def train(self, X_train, y_train):
        for epoch in range(self.epochs):
            outputs = self.forward(X_train.T)
            loss = self.mse_loss(outputs, y_train)
            self.losses.append(loss)
            self.update_parameters(self.backward(X_train, y_train))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat > 0.5).astype(int)

    def visualizing_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.show()


and_input = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

and_target = np.array([
    [0],
    [0],
    [0],
    [1]
])

input_size = and_input.shape[1]
hidden_sizes = [2, 2]
output_size = and_target.shape[1]
mlp = MLP(input_size, hidden_sizes, output_size, epochs=1000, learning_rate=0.01)

mlp.train(and_input, and_target)
mlp.visualizing_loss()
