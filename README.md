# IPL Score Predictor

This project predicts the final score of an IPL cricket match based on current match statistics like wickets fallen, current score, overs completed, and other relevant features.

## Project Structure

```
ipl_score_predictor/
├── data/              # Dataset storage
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks for exploration and visualization
├── src/               # Source code
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Collection and Preparation
- Use the scripts in the `src` directory to collect or prepare IPL match data
- Store raw and processed datasets in the `data` directory

### Model Training
- Notebooks in the `notebooks` directory showcase exploratory data analysis and model development
- Trained models are saved in the `models` directory

### Prediction
- Use the prediction scripts in the `src` directory to make score predictions
- The web application provides an intuitive interface for making predictions

## Models Used
- Linear Regression
- Random Forest
- XGBoost
- Neural Networks (LSTM)

## Web Interface
A user-friendly web interface is provided for making predictions without needing to understand the underlying model. 