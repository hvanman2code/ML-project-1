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


class IPLScorePredictor:
    """
    Class for predicting IPL match scores using trained models
    """
    
    def __init__(self, model_path=None, preprocessor=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model
            preprocessor: Fitted preprocessor for transforming input data
        """
        self.model = None
        self.preprocessor = preprocessor
        self.data_processor = IPLDataProcessor()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the model file
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_preprocessor(self, preprocessor_path):
        """
        Load a fitted preprocessor from disk
        
        Args:
            preprocessor_path: Path to the preprocessor file
        """
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor loaded from {preprocessor_path}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor: {str(e)}")
            return False
    
    def predict_score(self, match_features):
        """
        Predict the final score based on current match features
        
        Args:
            match_features: Dictionary or DataFrame with match features
            
        Returns:
            Predicted final score
        """
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return None
            
        # Convert dict to DataFrame if needed
        if isinstance(match_features, dict):
            match_features = pd.DataFrame([match_features])
            
        # Feature engineering if needed
        match_features = self.data_processor._create_features(match_features)
        
        # Transform features using preprocessor if available
        if self.preprocessor:
            X = self.preprocessor.transform(match_features)
        else:
            # If no preprocessor, use raw features (not recommended in production)
            X = match_features
            
        # Make prediction
        predicted_score = self.model.predict(X)
        
        return int(round(predicted_score[0]))
    
    def explain_prediction(self, match_features):
        """
        Provide an explanation for the prediction if possible
        
        Args:
            match_features: Dictionary or DataFrame with match features
            
        Returns:
            Dictionary with prediction and explanation
        """
        predicted_score = self.predict_score(match_features)
        
        if predicted_score is None:
            return {"error": "No prediction available"}
            
        # Basic explanation based on model type
        explanation = {}
        
        if hasattr(self.model, "feature_importances_"):
            # For tree-based models
            feature_importance = {}
            
            if isinstance(match_features, dict):
                features = list(match_features.keys())
            else:
                features = match_features.columns.tolist()
                
            for i, importance in enumerate(self.model.feature_importances_):
                if i < len(features):
                    feature_importance[features[i]] = float(importance)
                    
            # Sort by importance
            feature_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
            
            explanation["feature_importance"] = feature_importance
            
        elif hasattr(self.model, "coef_"):
            # For linear models
            coefficients = {}
            
            if isinstance(match_features, dict):
                features = list(match_features.keys())
            else:
                features = match_features.columns.tolist()
                
            for i, coef in enumerate(self.model.coef_):
                if i < len(features):
                    coefficients[features[i]] = float(coef)
                    
            explanation["coefficients"] = coefficients
            
        return {
            "predicted_score": predicted_score,
            "explanation": explanation
        }


if __name__ == "__main__":
    # Example usage
    import os
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    # Path to the best model (you may want to use a configuration file for this)
    model_path = os.path.join(models_dir, 'xgboost_model.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    
    # Create predictor and load model
    predictor = IPLScorePredictor()
    predictor.load_model(model_path)
    predictor.load_preprocessor(preprocessor_path)
    
    # Example match features
    match_features = {
        'runs': 120,  # Current score
        'wickets': 3,  # Wickets fallen
        'overs': 15.2,  # Overs completed
        'striker_runs': 45,  # Current batsman's score
        'non_striker_runs': 30,  # Non-striker's score
        'last_five_overs_runs': 42,  # Runs in last 5 overs
        'batting_team': 'Mumbai Indians',  # Batting team
        'bowling_team': 'Chennai Super Kings',  # Bowling team
        'venue': 'Wankhede Stadium'  # Match venue
    }
    
    # Make prediction
    prediction = predictor.predict_score(match_features)
    print(f"Predicted final score: {prediction}")
    
    # Get explanation
    explanation = predictor.explain_prediction(match_features)
    print("\nExplanation:")
    for key, value in explanation.get("explanation", {}).items():
        print(f"\n{key.title()}:")
        for feature, importance in value.items():
            print(f"  {feature}: {importance:.4f}") 