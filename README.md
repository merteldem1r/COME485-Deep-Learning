# Deep Learning - COME485

**Mert Eldemir** | COME485 - Deep Learning Course

---

## Topics Covered

### 1. Perceptron
**Purpose:** Understanding the fundamental **building block of neural networks**.

**Key Concepts:**
- Binary classification with **step activation** function
- **Weight updates** using learning rule: `w = w + η·e·x`
- **AND gate** implementation example

**Files:**
- [perceptron.py](./Perceptron/perceptron.py) - Python implementation
- [Perceptron.md](./Perceptron/Perceptron.md) - Complete explanations

**Libraries:** `numpy`

---

### 2. Gradient Descent
**Purpose:** Learning the **optimization algorithm** behind neural network training.

**Key Concepts:**
- **Sigmoid activation** function
- **Sum of Squared Errors** (SSE) loss
- Calculating **gradients**: ∂E/∂w and ∂E/∂b
- **Weight update** process with learning rate

**Files:**
- [gradient.py](./Gradient/gradient.py) - Batch gradient descent implementation
- [Gradient.md](./Gradient/Gradient.md) - Mathematical derivations and worked examples

**Libraries:** `numpy`

---

### 3. ANN for Handwritten Digit Recognition (MNIST)
**Purpose:** Building a **complete neural network for real-world image classification**,

**Key Concepts:**
- **Multilayer Perceptron** architecture (784→120→84→10)
- Training with **backpropagation**
- Preprocessing: **normalization**, **color inversion**, **thresholding**
- Model evaluation with **confusion matrix**

**Results:**
- Training accuracy: 99.24%
- Test accuracy: 97.59%
- Custom handwritten digits: 100% accuracy (for clear and bold written digits)

**Files:**
- [ann_for_mnist.ipynb](./ANN-Handwrite-Recognition/ann_for_mnist.ipynb) - Interactive Jupyter notebook
- [ANN-Handwrite.md](./ANN-Handwrite-Recognition/ANN-Handwrite.md) - Complete step-by-step explanation
- [digits-img/](./ANN-Handwrite-Recognition/digits-img/) - Custom handwritten test images

**Libraries:** `torch`, `torchvision`, `numpy`, `matplotlib`, `sklearn`, `seaborn`, `PIL`, `tqdm`

---

## Installation

```bash
pip install -r requirements.txt
```
