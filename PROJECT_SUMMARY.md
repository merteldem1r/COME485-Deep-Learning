# COME485 Deep Learning Project Summary

## Project Overview

I built a neural network using PyTorch to recognize handwritten digits from the MNIST dataset. The goal was to understand how deep learning works from the ground up, starting with basic concepts like perceptrons and gradient descent, then implementing a full neural network that could recognize my own handwritten digits.

## What I Did

### 1. Learning the Fundamentals
- Created documentation for **Perceptron** algorithm with step-by-step examples (AND gate implementation)
- Wrote comprehensive **Gradient Descent** explanation with mathematical derivations
- Studied how backpropagation works with sigmoid activation and squared error loss

### 2. Building the Neural Network
- Implemented a **Multilayer Perceptron** with PyTorch:
  - Input layer: 784 neurons (28×28 pixels)
  - Hidden layers: 120 and 84 neurons with ReLU activation
  - Output layer: 10 neurons (digits 0-9) with log_softmax
- Trained on MNIST dataset (60,000 training images, 10,000 test images)
- Used Adam optimizer with cross-entropy loss
- Training results: **99.24% train accuracy, 97.59% test accuracy** after 10 epochs

### 3. Testing with My Own Handwritten Digits
- Created 10 handwritten digit images (zero.png through nine.png)
- Modified the notebook to batch process all images from the `digits-img` folder
- Added visualization showing predictions with confidence scores

## Issues I Encountered

### Issue 1: Python 2 vs Python 3 Syntax Error
**Problem:** Code used `myiter.next()` which doesn't work in Python 3
**Solution:** Changed to `next(myiter)` - the Python 3 compatible syntax

### Issue 2: Missing Libraries
**Problem:** `sklearn` and `seaborn` weren't installed in my environment
**Solution:** Installed them with `pip install scikit-learn seaborn`

### Issue 3: **CRITICAL - All Predictions Were Wrong!**
**Problem:** The trained model (97.59% test accuracy) predicted **ALL** my handwritten digits as "3", even though they were clearly different digits. This was extremely frustrating because the model worked perfectly on MNIST data.

**Root Cause Discovery:** After adding diagnostic cells to compare MNIST images with my custom images, I found that:
- MNIST images have **pure black backgrounds** (pixel value = 0.0)
- My preprocessed images had **gray backgrounds** (pixel values = 0.27-0.29)
- Even after inverting colors, my images didn't match MNIST's binary nature

**The Solution:** Added a **thresholding step** after color inversion:
```python
threshold = 0.5
img_array = np.where(img_array > threshold, 1.0, 0.0)
```
This converts:
- Background pixels → 0.0 (pure black, like MNIST)
- Digit pixels → 1.0 (pure white, like MNIST)

**Result:** After applying the threshold and improving my handwriting (making digits bolder and clearer), I achieved **100% accuracy** on all 10 custom digits!

## Key Lessons Learned

1. **Preprocessing is Critical:** Even a perfectly trained model (97.59% accuracy) will fail if the input preprocessing doesn't match the training data format. Small differences in pixel distributions can completely break predictions.

2. **Debug Systematically:** When predictions fail, visualize and compare your data with the training data. The diagnostic cells I added were crucial for discovering the gray background problem.

3. **Match Training Conditions:** Neural networks learn patterns from training data. If your test images have different characteristics (like gray vs black backgrounds), the model won't generalize well.

4. **Accuracy is the Right Metric for MNIST:** Since the dataset is balanced (roughly equal samples per digit) and all misclassification errors are equally bad, accuracy is the most appropriate metric. Other metrics like F1-score, precision, or recall would be more important for imbalanced datasets or when certain errors are more costly than others.

## Final Results

- **MNIST Test Set:** 97.59% accuracy
- **Custom Handwritten Digits:** 100% accuracy (10/10 correct predictions)
- **Model Parameters:** 105,214 trainable parameters
- **Training Time:** ~10 minutes for 10 epochs

The project successfully demonstrated that with proper preprocessing and careful debugging, a simple neural network can achieve excellent results on both standard datasets and real-world handwritten images.
