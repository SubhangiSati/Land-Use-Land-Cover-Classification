# Land Use and Land Cover Classification

This project focuses on land use and land cover classification using Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The classification task aims to predict the category of land based on satellite or aerial images. The project involves the following components:

## Dataset

Two datasets are used for land classification:
- **EuroSAT Dataset**: A Sentinel-2 satellite image dataset covering various land use and land cover classes in Europe.

## CNN Model for Land Use Classification

### Overview
- The CNN model is trained on the EuroSAT dataset to classify land use and land cover categories.
- Data augmentation techniques such as rotation, shifting, and flipping are applied to enhance the model's robustness.

### Optimizers
- The model is trained with different optimizers including Adagrad.
- Hyperparameter tuning is performed to optimize the model's performance.

### Evaluation Metrics
- Metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate the model's performance.

### TensorBoard Integration
- TensorBoard is used for visualizing model training metrics such as accuracy, loss, precision, recall, and F1-score.

## RNN Model for Land Use Classification

### Overview
- The RNN model is trained on the EuroSAT dataset to classify land use and land cover categories.
- LSTM layers are utilized to capture sequential patterns in the image data.

### Optimizers
- The RNN model is trained using the Adagrad optimizer to optimize the learning process.

### Mixed Precision Training
- Mixed precision training is enabled to accelerate training and reduce memory usage.

### Evaluation Metrics
- Similar to the CNN model, precision, recall, and F1-score are computed to assess the RNN model's performance.

### TensorBoard Integration
- TensorBoard is utilized to visualize the training progress and performance metrics of the RNN model.

## Usage
1. **Data Preparation**: Ensure that the EuroSAT dataset is downloaded and preprocessed before training the models.
2. **Model Training**: Train the CNN and RNN models using the provided scripts.
3. **Evaluation**: Evaluate the models using appropriate evaluation metrics.
4. **Visualization**: Visualize the training progress and metrics using TensorBoard.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- scikit-learn
- Matplotlib (for visualization)

## Acknowledgments
- EuroSAT dataset source: [https://www.kaggle.com/datasets/apollo2506/eurosat-dataset]
- Special thanks to contributors Ayan Sar and Purvika Joshi for their valuable contributions.

