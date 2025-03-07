import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add the current directory to the path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data_processor import IPLDataProcessor
from src.model_trainer import IPLScoreModelTrainer
from src.generate_sample_data import generate_sample_data, save_data, train_test_split

def run_pipeline(regenerate_data=False, retrain_models=True):
    """
    Run the full IPL score prediction pipeline:
    1. Generate or load data
    2. Preprocess data
    3. Train models
    4. Evaluate models
    5. Save models
    
    Args:
        regenerate_data: Whether to regenerate the sample data
        retrain_models: Whether to retrain the models
    """
    print("=== IPL Score Predictor Pipeline ===")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    
    # Create directories if they don't exist
    for directory in [data_dir, models_dir, plots_dir]:
        os.makedirs(directory, exist_ok=True)
    
    raw_data_path = os.path.join(data_dir, 'ipl_raw_data.csv')
    processed_data_path = os.path.join(data_dir, 'ipl_processed_data.csv')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    
    # Step 1: Generate or load data
    if regenerate_data or not os.path.exists(raw_data_path):
        print("\nStep 1: Generating new sample data...")
        # Generate sample data
        sample_data = generate_sample_data(num_samples=1500)
        
        # Save the data
        save_data(sample_data, raw_data_path)
        print("Sample data generated and saved.")
    else:
        print("\nStep 1: Using existing data...")
        
    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    processor = IPLDataProcessor()
    if not os.path.exists(processed_data_path):
        # Preprocess the data
        data = processor.preprocess_data(raw_data_path)
        
        # Save the processed data
        processor.save_processed_data(data, processed_data_path)
        print("Data preprocessing completed.")
    else:
        print("Using existing processed data.")
        data = pd.read_csv(processed_data_path)
    
    # Step 3: Train models
    if retrain_models:
        print("\nStep 3: Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(data)
        
        # Create preprocessor
        preprocessor = processor.create_preprocessor()
        
        # Fit and transform data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save preprocessor
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
        
        # Train models
        trainer = IPLScoreModelTrainer(processor)
        trainer.train_models(X_train_processed, y_train)
        
        # Step 4: Evaluate models
        print("\nStep 4: Evaluating models...")
        results = trainer.evaluate_models(X_test_processed, y_test)
        
        # Plot results
        trainer.plot_results(save_dir=plots_dir)
        
        # Step 5: Save models
        print("\nStep 5: Saving models...")
        trainer.save_models(models_dir)
        
        # Print best model
        best_model_name, best_model = trainer.get_best_model()
        print(f"Best model: {best_model_name}")
    else:
        print("\nSkipping model training as requested.")
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the IPL score prediction pipeline')
    parser.add_argument('--regenerate-data', action='store_true', help='Regenerate sample data')
    parser.add_argument('--no-retrain', action='store_true', help='Skip model retraining')
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_pipeline(regenerate_data=args.regenerate_data, retrain_models=not args.no_retrain) 