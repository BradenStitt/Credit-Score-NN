# Credit Score Prediction System

## Overview

This project implements a machine learning model for predicting credit scores using a neural network approach as part of CSUB's CMPS 3500 Final Project. The system provides an interactive CLI for data loading, processing, model building, and testing.

## Features

- Interactive Command-Line Interface (CLI)
- Data Loading and Preprocessing
- Neural Network Model for Credit Score Prediction
- Performance Metrics Calculation
- CSV Output of Predictions

## Prerequisites

### Python Dependencies

- pandas
- numpy
- scikit-learn
- tensorflow
- keras
- tabulate
- csv

### Recommended Environment

- Python 3.8+
- Anaconda/Miniconda (Optional but recommended)

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd credit-score-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install requirements.txt
```

## Project Structure

```
project-root/
│
├── code/
│   └── menu.py
│
├── data/
│   └── credit_score_data.csv
│
└── README.md
```

## Data Requirements

- Input data should be a CSV file named `credit_score_data.csv`

## Usage

Run the script:

```bash
python menu.py
```

### Menu Options

1. **Load Data**:

   - Reads credit score dataset
   - Displays total columns and rows
   - Tracks data loading time

2. **Process Data**:

   - Performs comprehensive data cleaning
   - Handles missing values
   - Transforms categorical variables
   - Prepares data for model training

3. **Model Details**:

   - Builds neural network architecture
   - Configures model hyperparameters
   - Displays network structure

4. **Test Model**:

   - Trains neural network
   - Evaluates model performance
   - Generates predictions
   - Calculates performance metrics
   - Saves predictions to `predictions.csv`

5. **Quit**: Exit the application

## Model Architecture

### Neural Network Configuration

- Input Layer: 24 nodes
- Hidden Layers:
  - First Layer: 48 nodes (ReLU)
  - Second Layer: 96 nodes (ReLU)
  - Third Layer: 96 nodes (ReLU)
  - Fourth Layer: 48 nodes (ReLU)
- Output Layer: 3 nodes (Softmax)

### Training Parameters

- Loss Function: Categorical Cross-Entropy
- Epochs: 12
- Test Set Size: 20%

## Output

### Predictions CSV

- Columns: ID, Prediction
- Contains model's credit score predictions

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request
