# Animal Faces Classification: A Deep Dive into Image Recognition with PyTorch

This project offers a comprehensive walkthrough of building a Convolutional Neural Network (CNN) for image classification using PyTorch. We tackle the "Animal Faces" dataset from Kaggle, which contains images of cats, dogs, and wild animals, and train a model to distinguish between them with high accuracy.

This repository is not just about the final model, but about the journey: from data loading and preprocessing to model architecture, training, and evaluation. The entire process is documented in the `main.ipynb` Jupyter Notebook.

## Project Structure

```
.
├── data/
│   └── animal-faces/
│       ├── train/
│       │   ├── cat/
│       │   ├── dog/
│       │   └── wild/
│       └── val/
│           ├── cat/
│           ├── dog/
│           └── wild/
├── .gitignore
├── environment.yml
├── main.ipynb
├── README.md
└── requirements.txt
```

*   `data/`: Contains the image dataset, split into training and validation sets.
*   `environment.yml`: The Conda environment file.
*   `main.ipynb`: The heart of the project, a Jupyter Notebook with the complete workflow.
*   `README.md`: This file.
*   `requirements.txt`: The pip requirements file.

## Getting Started

### Prerequisites

*   Python 3.11
*   Conda (recommended) or pip

### Installation

There are two ways to set up the environment:

**1. Using Conda (recommended)**

This will create a new Conda environment with all the necessary dependencies.

1.  Create the Conda environment:
    ```bash
    conda env create -f environment.yml
    ```
2.  Activate the environment:
    ```bash
    conda activate animal-faces-classification
    ```

**2. Using pip**

1.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How It Works: A Look Inside `main.ipynb`

The `main.ipynb` notebook is structured to guide you through the entire process of building an image classification model.

### 1. Data Exploration and Preprocessing

*   **Loading the Data:** We start by loading the image paths and their corresponding labels into a Pandas DataFrame.
*   **Data Splitting:** The dataset is split into training, validation, and test sets.
*   **Transformations:** We define a series of transformations (resizing, converting to tensor) to be applied to the images before they are fed into the model.
*   **Custom Dataset:** A custom `AnimalFaceDataset` class is created to handle the loading and transformation of the images.
*   **DataLoaders:** We create `DataLoaders` for the training, validation, and test sets to efficiently load the data in batches.

### 2. Building the Model

The model is a custom-built Convolutional Neural Network (CNN) with the following architecture:

*   Conv2d(3, 32, kernel_size=3, padding=1)
*   MaxPool2d(2, 2)
*   ReLU
*   Conv2d(32, 64, kernel_size=3, padding=1)
*   MaxPool2d(2, 2)
*   ReLU
*   Conv2d(64, 128, kernel_size=3, padding=1)
*   MaxPool2d(2, 2)
*   ReLU
*   Flatten
*   Linear(128 * 16 * 16, 128)
*   Linear(128, 3)

### 3. Training and Validation

*   **Loss Function and Optimizer:** We use `CrossEntropyLoss` as the loss function and the `Adam` optimizer.
*   **Training Loop:** The model is trained for a set number of epochs. In each epoch, we:
    *   Feed the training data through the model.
    *   Calculate the loss.
    *   Perform backpropagation to update the model's weights.
    *   Calculate the training accuracy.
*   **Validation Loop:** After each epoch, we evaluate the model's performance on the validation set to monitor for overfitting.
*   **Plotting Results:** The training and validation loss and accuracy are plotted to visualize the model's learning progress.

### 4. Inference

The notebook includes a `predict_image` function to classify new images. This function:

1.  Opens and transforms the image.
2.  Feeds it through the trained model.
3.  Returns the predicted label.

To use it, simply place your image in the `data/animal-faces/` directory and update the file path in the function call.

## Dependencies

This project uses the following libraries:

*   [Python 3.11](https://www.python.org/)
*   [PyTorch](https://pytorch.org/) (>=2.1, <2.3)
*   [torchvision](https://pytorch.org/vision/stable/index.html)
*   [NumPy](https://numpy.org/) (<2.0)
*   [Pandas](https://pandas.pydata.org/)
*   [Matplotlib](https://matplotlib.org/)
*   [scikit-learn](https://scikit-learn.org/stable/)
*   [Kaggle](https://www.kaggle.com/docs/api)
*   [opendatasets](https://github.com/JovianML/opendatasets)
*   [torchinfo](https://github.com/TylerYep/torchinfo)

You can find the specific versions in `requirements.txt` and `environment.yml`.
