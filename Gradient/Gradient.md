# Gradient Descent 

**NOTE**: Explanations made with AI

## What is Gradient Descent?

Gradient descent is an **optimization algorithm** used to find the best parameters (weights and bias) for a neural network model by minimizing an error function. Think of it like descending a hill in the fog: you can't see the bottom, but you can feel which direction is downhill. You take a step downhill, feel the slope again, and repeat until you reach the valley.

The algorithm works by:
1. Computing how much each parameter contributes to the error (the **gradient**)
2. Taking a small step in the opposite direction of the gradient (downhill)
3. Repeating until the error is minimized or a stopping condition is met

## Key Components

### 1. The Sigmoid Function

The sigmoid function is a smooth, S-shaped curve that squashes any input into a value between 0 and 1. It's commonly used as an **activation function** in neural networks.

$$f(x) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

Breaking this down:
- **$w \cdot x + b$** = the weighted sum (linear part)
- **$e^{-(w \cdot x + b)}$** = exponential, makes negative inputs very large
- **$\frac{1}{1 + \text{large number}}$** = divides by a large number to squeeze output to (0, 1)

**In code:**
```python
def f(w, b, x):  # sigmoid with parameters w,b
    return 1.0 / (1.0 + np.exp(-(w*x + b)))
```

**Example:** If $w = 1, b = 0, x = 0$:
- $f(0) = \frac{1}{1 + e^{0}} = \frac{1}{1 + 1} = 0.5$

If $x = 2$ (larger input):
- $f(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.135} \approx 0.88$ (closer to 1)

**Why sigmoid?** It's smooth and differentiable, which means we can compute gradients everywhere. The perceptron used a hard step function, which is not differentiable; sigmoid allows gradient-based learning.

### 2. The Error Function (Sum of Squared Errors)

We need a way to measure how bad our predictions are. The **Sum of Squared Errors (SSE)** does this:

$$E(w, b) = \sum_{i} \frac{1}{2} (f(x_i) - y_i)^2$$

Where:
- **$f(x_i)$** = predicted output from the sigmoid
- **$y_i$** = actual (desired) output
- **$(f(x_i) - y_i)$** = prediction error for that sample
- **$(f(x_i) - y_i)^2$** = squared error (makes all errors positive)
- **$\sum$** = sum over all training examples

**Interpretation:** If prediction matches truth ($f(x_i) = y_i$), error is 0. If they differ, error grows quadratically (large errors are penalized heavily).

**In code:**
```python
def error(w, b):  # SSE error function
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        err += 0.5 * (fx - y)**2
    return err
```

### 3. Partial Derivatives (Gradients)

The **gradient** tells us: "If I change parameter $w$ (or $b$) slightly, how much does the error change?"

For the sigmoid with SSE, the partial derivatives are:

$$\frac{\partial E}{\partial b} = (f(x) - y) \cdot f(x) \cdot (1 - f(x))$$

$$\frac{\partial E}{\partial w} = (f(x) - y) \cdot f(x) \cdot (1 - f(x)) \cdot x$$

**Breaking down the first one:**
- **(f(x) - y)** = prediction error (how far off we are)
- **f(x)** = sigmoid output (between 0 and 1)
- **(1 - f(x))** = "steepness" of the sigmoid at this point

**In code:**
```python
def grad_b(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)

def grad_w(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x
```

**Key insight:** The gradient for $w$ includes an extra factor of $x$ because changing $w$ is "scaled" by the input value.

### 4. The Gradient Descent Update Rule

Once we know the gradient, we update parameters by moving opposite to it:

$$w := w - \eta \cdot \frac{\partial E}{\partial w}$$
$$b := b - \eta \cdot \frac{\partial E}{\partial b}$$

Where:
- **$\eta$** (eta) = learning rate (controls step size; e.g., 0.1, 1.0)
- **$\frac{\partial E}{\partial w}$** = gradient (slope of error function)
- **$:=$** = "update to" (assignment operator)

**Intuition:** If the gradient is positive (error increases as $w$ increases), we decrease $w$. If gradient is negative (error decreases as $w$ increases), we increase $w$. The learning rate controls how big each step is.

## The Algorithm: Gradient Descent in Steps

1. **Initialize** parameters $w$ and $b$ (usually to small random values or zeros)
2. **For each epoch** (pass through the data):
   a. **For each training sample** $(x_i, y_i)$:
      - Compute gradients: $\frac{\partial E}{\partial w}$ and $\frac{\partial E}{\partial b}$
      - Accumulate them (sum up contributions from all samples)
   b. **Update parameters**:
      - $w := w - \eta \cdot (\text{sum of } \frac{\partial E}{\partial w})$
      - $b := b - \eta \cdot (\text{sum of } \frac{\partial E}{\partial b})$
3. **Check stopping condition**: Stop if error is small enough, or max epochs reached

**In code:**
```python
def do_gradient_descent():
    w, b, eta, max_epochs = -2.0, 0.0, 1.0, 1000
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)  # accumulate gradients
            db += grad_b(w, b, x, y)
        w = w - eta * dw  # update w
        b = b - eta * db  # update b
        print(f"Epoch {i+1}: w = {w}, b = {b}, error = {error(w, b)}")
```

## Worked Example: Step by Step

Let's trace through the first few iterations with the data in `gradient.py`:

**Initial data:**
- Training examples: $X = [0.5, 2.5]$, $Y = [0.2, 0.9]$
- Initial parameters: $w = -2.0$, $b = 0.0$
- Learning rate: $\eta = 1.0$

### Epoch 1

**For sample $(x_1, y_1) = (0.5, 0.2)$:**

1. **Compute sigmoid prediction:**
   - $z = w \cdot x + b = -2.0 \cdot 0.5 + 0.0 = -1.0$
   - $f(0.5) = \frac{1}{1 + e^{1.0}} = \frac{1}{1 + 2.718} \approx 0.269$
   - We predicted 0.269, but truth is 0.2 (prediction is too high)

2. **Compute gradients:**
   - $\frac{\partial E}{\partial b} = (0.269 - 0.2) \cdot 0.269 \cdot (1 - 0.269) = 0.069 \cdot 0.269 \cdot 0.731 \approx 0.0136$
   - $\frac{\partial E}{\partial w} = 0.0136 \cdot 0.5 \approx 0.0068$

So the gradients are: $db_1 = 0.0136$, $dw_1 = 0.0068$

**For sample $(x_2, y_2) = (2.5, 0.9)$:**

1. **Compute sigmoid prediction:**
   - $z = -2.0 \cdot 2.5 + 0.0 = -5.0$
   - $f(2.5) = \frac{1}{1 + e^{5.0}} = \frac{1}{1 + 148.4} \approx 0.0067$
   - We predicted 0.0067, but truth is 0.9 (prediction is way too low!)

2. **Compute gradients:**
   - Error $(f - y) = 0.0067 - 0.9 = -0.893$ (large negative error)
   - $\frac{\partial E}{\partial b} = -0.893 \cdot 0.0067 \cdot (1 - 0.0067) \approx -0.00598$
   - $\frac{\partial E}{\partial w} = -0.00598 \cdot 2.5 \approx -0.01496$

So the gradients are: $db_2 = -0.00598$, $dw_2 = -0.01496$

**Accumulate and update:**

- Total gradients: $dw_{total} = 0.0068 + (-0.01496) = -0.00816$, $db_{total} = 0.0136 + (-0.00598) = 0.00762$
- Update $w$: $w := -2.0 - 1.0 \cdot (-0.00816) = -2.0 + 0.00816 = -1.99184$
- Update $b$: $b := 0.0 - 1.0 \cdot 0.00762 = -0.00762$

**After Epoch 1:** $w \approx -1.99184$, $b \approx -0.00762$

### What's happening?

The algorithm is **learning**:
- The first sample said "prediction is slightly too high," so we decrease $w$ and $b$ (but barely).
- The second sample said "prediction is way too low," which has a strong opposing gradient.
- These competing signals partially cancel out, resulting in small updates.
- Over many epochs, $w$ will increase (become less negative) so that $x = 2.5$ produces a higher output closer to 0.9.

## Why Gradient Descent Works

1. **Direction:** The gradient points in the direction of steepest increase of error. Moving opposite to it reduces error.
2. **Smoothness:** The sigmoid is smooth (differentiable everywhere), so gradients exist and guide us accurately.
3. **Convergence:** For convex problems, gradient descent will find the global minimum.
4. **Efficiency:** Each update is computationally simple and moves us toward the solution.

## Learning Rate ($\eta$) Matters

- **Too small $\eta$ (e.g., 0.001):** Very slow convergence; many epochs needed.
- **Just right (e.g., 0.1 or 1.0):** Fast convergence, weights update meaningfully each epoch.
- **Too large $\eta$ (e.g., 10.0):** May overshoot the minimum and diverge (error increases instead of decreases).

## Limitations and Extensions

- **Local minima:** In non-convex problems (like deep neural nets), gradient descent may get stuck in local minima instead of finding the global minimum.
- **Scalability:** Computing gradients for all samples (batch gradient descent, as in this code) is slow for large datasets.
- **Solution:** Use **mini-batch** or **stochastic** gradient descent: update after each sample or small batch instead of all data.

## Summary

Gradient descent is the workhorse of neural network training:
1. **Compute error** using a loss function (SSE in this case)
2. **Compute gradients** (partial derivatives) to see which direction reduces error
3. **Update parameters** by small steps in the opposite direction of gradients
4. **Repeat** until convergence

The code in `gradient.py` implements batch gradient descent on a simple 2-sample dataset with a sigmoid neuron, learning weights that minimize prediction error.
