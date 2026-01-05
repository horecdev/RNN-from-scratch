# RNN-from-Scratch

Implementation of a Recurrent Neural Network **from scratch** in NumPy, using just math

## Features

- Manually written forward pass and Backpropagation Through Time `(BPTT)`
- Full model and layers implemented from scratch:
    - `RNN`
    - `Embedding`
    - `SoftmaxCrossEntropy`
    - `MSELoss`
- Simple SGD param update
- Stacking RNN models on one another to boost performance

## Experiments

### 1. Sine Wave Sanity Check
- File: `sine.ipynb`
- A sanity check to ensure that BPTT works
- Proof that model can remember its past states to make predictions

### 2. Text Generation (Sherlock Holmes)
- File: `text_gen.ipynb`
- Trains a model to generate continous text based on Sherock Holmes book
- Achieves loss of 1.50 and outputs sentences that are similar to english, somewhat correctly uses punctuation
- Shows that simple RNN performs decently in NLP (I want to have a comparison before I move on to LSTMs)

### 3. Three-Body Gravity Simulation
- File: `gravity_sim`
- Trains a model to learn laws of gravity `G * m1 * m2 / r^2` on a system consisting of three objects
- Shows some level of understanding of the "smoothness" of orbits, creates smooth simulation, but isn't perfect
