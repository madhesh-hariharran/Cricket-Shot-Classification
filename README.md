# Cricket Shot Classification

## Overview
This project implements a deep learning-based cricket shot classification system using multiple convolutional neural network (CNN) architectures. The goal is to classify different cricket shots based on video frames using deep learning models.

## Dataset
The dataset used for training and evaluation is the [Cricket Shot Dataset](https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset) from Kaggle. It contains labeled images of various cricket shots.

## Models Implemented
The following CNN architectures have been implemented and tested:
- **DenseNet** (`DENSENet.ipynb`)
- **EfficientNet** (`EFFICIENTNet.ipynb`)
- **InceptionNet** (`INCEPTIONNet.ipynb`)
- **ResNet** (`RESNet.ipynb`)
- **SENet** (`SENet.ipynb`)

## Requirements
To run the models, install the following dependencies:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cricket-shot-classification.git
   cd cricket-shot-classification
   ```
2. Download and extract the dataset from the Kaggle link.
3. Open and run any of the Jupyter notebooks (`.ipynb`) to train and test the models.

## License
This project is licensed under the MIT License.
