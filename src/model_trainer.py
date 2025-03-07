import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Add the current directory to the path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data_processor import IPLDataProcessor

class IPLScoreModelTrainer:
    """
    Class for training various models to predict IPL scores
    """
    
    def __init__(self, data_processor=None):
        """
        Initialize the model trainer
        
        Args:
            data_processor: Instance of IPLDataProcessor class
        """
        self.data_processor = data_processor if data_processor else IPLDataProcessor()
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        self.trained_models = {}
        self.results = {}
        
    def train_models(self, X_train, y_train):
        """
        Train all models on the provided data
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"{name} training completed")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and return their performance metrics
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of performance metrics for each model
        """
        results = {}
        
        for name, model in self.trained_models.items():
            print(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
        self.results = results
        return results
    
    def save_models(self, models_dir):
        """
        Save all trained models to disk
        
        Args:
            models_dir: Directory to save the models
        """
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            model_path = os.path.join(models_dir, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"Model {name} saved to {model_path}")
    
    def plot_results(self, results=None, save_dir=None):
        """
        Plot the performance of all models
        
        Args:
            results: Dictionary of model results (optional)
            save_dir: Directory to save the plots (optional)
        """
        results = results if results else self.results
        
        if not results:
            print("No results to plot. Evaluate models first.")
            return
            
        # Create metrics comparison bar chart
        metrics = ['rmse', 'mae']
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.trained_models))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in results.keys()]
            ax.bar(x + i*width, values, width, label=metric.upper())
        
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(list(results.keys()))
        ax.legend()
        
        # Add R² values as text
        for i, model in enumerate(results.keys()):
            ax.text(i, 5, f"R² = {results[model]['r2']:.2f}", ha='center')
        
        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
            print(f"Plot saved to {os.path.join(save_dir, 'model_comparison.png')}")
        else:
            plt.show()
            
    def get_best_model(self, metric='rmse'):
        """
        Return the best model based on the specified metric
        
        Args:
            metric: Metric to use for comparison (default: 'rmse')
            
        Returns:
            Name of the best model and the model object
        """
        if not self.results:
            print("No results available. Evaluate models first.")
            return None, None
            
        if metric in ['rmse', 'mse', 'mae']:
            # Lower is better
            best_model_name = min(self.results, key=lambda m: self.results[m][metric])
        else:  # r2
            # Higher is better
            best_model_name = max(self.results, key=lambda m: self.results[m][metric])
            
        return best_model_name, self.trained_models[best_model_name]


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    
    processed_data_path = os.path.join(data_dir, 'ipl_processed_data.csv')
    
    # Check if the processed data exists
    if os.path.exists(processed_data_path):
        # Load and prepare data
        processor = IPLDataProcessor()
        data = pd.read_csv(processed_data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(data)
        
        # Create preprocessor
        preprocessor = processor.create_preprocessor()
        
        # Transform data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train models
        trainer = IPLScoreModelTrainer(processor)
        trainer.train_models(X_train_processed, y_train)
        
        # Evaluate models
        results = trainer.evaluate_models(X_test_processed, y_test)
        
        # Plot results
        trainer.plot_results(save_dir=plots_dir)
        
        # Save models
        trainer.save_models(models_dir)
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model()
        print(f"Best model: {best_model_name}")
    else:
        print(f"Processed data file not found at {processed_data_path}")
        print("Please run data_processor.py first to prepare the data") 