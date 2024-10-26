# MLP Classifier and Adversarial Attack Implementation

This project implements an MLP (Multi-Layer Perceptron) classifier and an adversarial attack strategy. Additionally, it includes a robust version of the classifier designed to withstand adversarial attacks.

## Project Structure

### Files

- **`classifier.py`**: Trains the classifier and saves the best model as `mlp_final.pt` in the `checkpoints` folder. The script automatically loads this checkpoint (the best classifier) and prints accuracy metrics after training.
  - **Running Instructions**:
    - No specific setup is required. The default parameters have been adjusted to match our final configuration.
    - Note: The script depends on other files in the `src` folder and cannot run as a standalone script without its environment.

- **`attack.py`**: Implements the adversarial attack. It loads the `mlp_final.pt` file and runs the attack across all bounds, printing the attack results.
  - **Running Instructions**:
    - Ensure that the `classifier.py` script has been executed first, as it generates the trained model (`mlp_final.pt`) required by this script.
    - The script depends on other files in the `src` folder and cannot run standalone.

- **`bonus.py`**: Implements a robust classifier. This script trains the robust classifier and saves the best model as `robust_mlp_final.pt` in the `checkpoints` folder. It then automatically loads the best classifier, prints accuracy metrics, and runs the attack on the classifier, printing the results.
  - **Running Instructions**:
    - No specific setup is required. The default parameters have been adjusted to match our final configuration.
    - The script depends on other files in the `src` folder and cannot run standalone.

## Folder Structure

- **`src/`**
  - **`mlp.py`**: Implementation of a configurable MLP network based on the implementation provided in HW2.
  - **`plot.py`**: Helper functions for plotting training results, adapted from previous homework.
  - **`train_results.py`**: Defines several data classes used to store and handle training results.
  - **`training.py`**: Defines the training process, including the classifier trainer and adversarial trainer implementations, based on previous homework.
  - **`tuneMlpScript.py`**: A script used to tune the MLP classifier, as described in the project report.
  - **`SIREN.py`** & **`utils.py`**: Provided files, not modified as part of this project.

## Jupyter Notebooks

- **`Part1.ipynb`**: Contains the code used for training the classifier, along with code for plotting the confusion matrix.
- **`Part2.ipynb`**: Contains the code used for the attack implementation, including functions for plotting the confusion matrix and requested images.
- **`Bonus.ipynb`**: Contains the code used for training and attacking the robust classifier, along with code for plotting the requested images.
