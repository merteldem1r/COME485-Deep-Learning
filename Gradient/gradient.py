import time
import numpy as np

X = [0.5, 2.5]  # input data points
Y = [0.2, 0.9]  # desired output data points


def f(w, b, x):  # sigmoid with parameters w,b (here we produce predicted output for xi)
    return 1.0 / (1.0 + np.exp(-(w*x + b)))


def error(w, b):  # Sum of Squared Errors function
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        err += 0.5 * (fx - y)**2
    return err


# Partial Derivatives (Gradients) for Bias and Weight update
def grad_b(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)


def grad_w(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


# The Gradient Descent Update Rule
def do_gradient_descent():
    # n = eta = learning rate
    w, b, eta, max_epochs = -2.0, 0.0, 1.0, 1000
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        w = w - eta * dw
        b = b - eta * db

        time.sleep(0.25)  # Slow down for visualization purposes
        print(f"Epoch {i+1}: w = {w}, b = {b}, error = {error(w, b)}")


if __name__ == "__main__":
    do_gradient_descent()
