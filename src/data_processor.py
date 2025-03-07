import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class IPLDataProcessor:
    """
    Class for processing IPL match data for model training and prediction
    """
    
    def __init__(self):
        self.numerical_features = ['runs', 'wickets', 'overs', 'striker_runs', 
                                  'non_striker_runs', 'last_five_overs_runs']
        self.categorical_features = ['batting_team', 'bowling_team', 'venue']
        self.target = 'final_score'
        self.preprocessor = None
        
    def preprocess_data(self, data_path):
        """
        Preprocess the raw IPL data
        
        Args:
            data_path: Path to the CSV file containing IPL match data
            
        Returns:
            Processed DataFrame ready for model training
        """
        # Read the data
        data = pd.read_csv(data_path)
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Feature engineering
        data = self._create_features(data)
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # Fill numerical missing values with median
        for col in self.numerical_features:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        # Fill categorical missing values with mode
        for col in self.categorical_features:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])
                
        return data
    
    def _create_features(self, data):
        """Create new features from existing ones"""
        if 'runs' in data.columns and 'overs' in data.columns:
            # Calculate run rate
            data['run_rate'] = data['runs'] / data['overs']
            self.numerical_features.append('run_rate')
            
        if 'wickets' in data.columns:
            # Calculate wickets in hand
            data['wickets_in_hand'] = 10 - data['wickets']
            self.numerical_features.append('wickets_in_hand')
            
        return data
    
    def create_preprocessor(self):
        """Create a scikit-learn preprocessor pipeline for the data"""
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor
    
    def split_data(self, data, test_size=0.2, random_state=42):
        """
        Split the data into features and target, and then into train and test sets
        
        Args:
            data: Preprocessed DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        X = data[self.numerical_features + self.categorical_features]
        y = data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, data, output_path):
        """Save the processed data to a CSV file"""
        data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import os
    
    # Create processor
    processor = IPLDataProcessor()
    
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_data_path = os.path.join(data_dir, 'ipl_raw_data.csv')
    processed_data_path = os.path.join(data_dir, 'ipl_processed_data.csv')
    
    # Check if the raw data exists (this is for demonstration - you'll need to provide actual data)
    if os.path.exists(raw_data_path):
        # Preprocess the data
        processed_data = processor.preprocess_data(raw_data_path)
        
        # Save the processed data
        processor.save_processed_data(processed_data, processed_data_path)
    else:
        print(f"Raw data file not found at {raw_data_path}")
        print("Please download IPL match data and place it in the data directory") 