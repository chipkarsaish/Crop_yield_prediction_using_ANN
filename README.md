
# Crop Yield Prediction using ANN

This project implements an end-to-end regression model to predict crop yields (measured in hectograms per hectare) using various factors like location, crop type, year, rainfall, pesticide usage, and average temperature. It includes a data preprocessing pipeline, model training scripts, and a Streamlit-based web interface for real-time predictions.

## Model Specifications

The predictive model is built using a **Sequential Artificial Neural Network (ANN)** implemented with TensorFlow/Keras. Below are the detailed specifications:

* **Model Architecture**:
* **Input Layer**: Accepts 15 input features (after preprocessing and encoding).
* **Hidden Layer 1**: 64 neurons with ReLU (Rectified Linear Unit) activation function.
* **Hidden Layer 2**: 32 neurons with ReLU activation function.
* **Output Layer**: 1 neuron (for regression) to predict the numerical yield value.


* **Compilation Details**:
* **Optimizer**: Adam.
* **Loss Function**: Mean Absolute Error (MAE).
* **Metrics**: Mean Absolute Error (MAE).


* **Training Configuration**:
* **Epochs**: 100.
* **Callbacks**: Includes Early Stopping (patience of 15 epochs) to prevent overfitting and TensorBoard for visualization.
* **Data Split**: 80% Training, 20% Testing.


* **Performance**: The model achieved a validation MAE of approximately 17,074 hg/ha after 100 epochs.

## Dataset and Features

The model is trained on a dataset (`yield.csv`) containing the following features:

| Feature Name | Type | Description |
| --- | --- | --- |
| **Area** | Categorical | Country or region name (e.g., India, Brazil). |
| **Item** | Categorical | Crop type (e.g., Maize, Rice, Wheat). |
| **Year** | Temporal | Year of observation. |
| **Average Rainfall** | Numerical | Annual rainfall in mm per year. |
| **Pesticides** | Numerical | Amount of pesticides used in tonnes. |
| **Average Temp** | Numerical | Average yearly temperature in °C. |
| **hg/ha_yield** | **Target** | Crop yield in hectograms per hectare. |

## Preprocessing Pipeline

To prepare the data for the ANN, the following steps are performed:

1. **Encoding**:
* **Item**: One-Hot Encoded (transformed into 10 binary columns).
* **Area**: Target Encoded (using `category_encoders.TargetEncoder` to handle high cardinality).


2. **Scaling**: All numerical features are standardized using `StandardScaler` to ensure the neural network converges efficiently.
3. **Artifacts**: The trained encoders and scalers are saved as `.pkl` files in the `artifacts/` folder for use during inference.

## Project Structure

* `app.py`: The Streamlit web application providing the user interface.
* `cleaning.ipynb`: Jupyter notebook detailing data cleaning, encoding, scaling, and model training.
* `predict.ipynb`: Notebook containing the logic for loading the model and making test predictions.
* `model.h5`: The saved trained ANN model.
* `artifacts/`: Contains `Age_encoder.pkl` (Area encoder), `onehot_encoder.pkl`, and `scaler.pkl`.
* `requirements.txt`: Lists all necessary Python libraries (TensorFlow, Streamlit, Pandas, etc.).

## Installation and Usage

1. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


2. **Run the Web Application**:
```bash
streamlit run app.py

```


3. **Prediction**: Enter the required crop and environmental details in the Streamlit UI to receive an instant yield prediction.
