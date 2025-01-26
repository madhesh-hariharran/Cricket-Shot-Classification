# Cricket Shot Classification

## Overview
This project focuses on developing a deep learning-based cricket shot classification system using multiple state-of-the-art convolutional neural network (CNN) architectures. The primary objective is to classify different cricket shots from video frames and analyze their effectiveness in recognizing and distinguishing between various batting techniques. 

The models implemented in this repository leverage fine-tuning of pre-existing architectures to achieve optimal performance on the dataset. By experimenting with different CNN architectures, we evaluate the efficiency, accuracy, and computational complexity of each model.

## Dataset
The dataset used for training and evaluation is the [Cricket Shot Dataset](https://www.kaggle.com/datasets/aneesh10/cricket-shot-dataset) from Kaggle. This dataset consists of labeled images representing different types of cricket shots, such as cover drive, pull shot, and cut shot. The dataset is preprocessed to ensure uniformity and optimal feature extraction.

### Preprocessing Steps:
- **Data Augmentation**: Techniques like rotation, flipping, and normalization are applied to improve generalization.
- **Image Resizing**: Standardized to a fixed dimension suitable for input to deep learning models.
- **Label Encoding**: Convert categorical shot labels into numerical representations for training.

## Models Implemented
The following CNN architectures have been implemented and evaluated:

### 1. **DenseNet** (`DENSENet.ipynb`)
- Uses densely connected convolutional layers to improve feature propagation and reduce vanishing gradients.
- Requires a low number of parameters compared to traditional CNN architectures.

### 2. **EfficientNet** (`EFFICIENTNet.ipynb`)
- Scales depth, width, and resolution efficiently to achieve high accuracy with minimal computational cost.
- Implemented with pre-trained weights and fine-tuned for cricket shot classification.
- Has a moderate number of parameters.

### 3. **InceptionNet** (`INCEPTIONNet.ipynb`)
- Uses multi-scale feature extraction by combining multiple convolutional kernel sizes in a single layer.
- Helps capture fine-grained details in cricket shot images.
- Has a high number of parameters.

### 4. **ResNet** (`RESNet.ipynb`)
- Incorporates residual connections to enable deep networks without degradation issues.
- Evaluated with different ResNet variants for optimal accuracy.
- Requires a moderate number of parameters.

### 5. **SENet** (`SENet.ipynb`)
- Implements Squeeze-and-Excitation (SE) blocks to recalibrate channel-wise feature responses.
- Helps focus on important regions of cricket shots to improve classification accuracy.
- Has a high number of parameters.

## Requirements
To run the models, install the following dependencies:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python
```

## Training and Evaluation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cricket-shot-classification.git
   cd cricket-shot-classification
   ```
2. Download and extract the dataset from Kaggle.
3. Load any of the Jupyter notebooks (`.ipynb`) and execute the training pipeline.
4. Hyperparameter tuning is applied using learning rate schedulers, batch normalization, and dropout layers.
5. The models are evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Results and Comparison
| Model      | Accuracy | Parameters | Computational Cost |
|------------|----------|------------|--------------------|
| DenseNet121   | 97.66%   | Low        | Moderate          |
| EfficientNetB0 | 96.36% | Moderate   | High              |
| InceptionV3 | 97.48% | High       | High              |
| ResNet50     | 93.99%  | Moderate   | Moderate         |
| SENet50      | 96.76%  | High       | High              |

## Future Work
- Exploring Transformer-based architectures for improved classification.
- Integrating temporal sequence models like LSTMs or GRUs for video-based cricket shot analysis.
- Deploying the trained model as a web application for real-time classification.

## License
This project is licensed under the MIT License.
