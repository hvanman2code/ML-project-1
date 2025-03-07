import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Add the parent directory to the path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.predictor import IPLScorePredictor
from src.data_processor import IPLDataProcessor

# Page configuration
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Define teams and venues
IPL_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

IPL_VENUES = [
    "Wankhede Stadium",
    "M. A. Chidambaram Stadium",
    "Eden Gardens",
    "Arun Jaitley Stadium",
    "M. Chinnaswamy Stadium",
    "Narendra Modi Stadium",
    "Rajiv Gandhi International Cricket Stadium",
    "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium",
    "Dr. DY Patil Sports Academy"
]

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    preprocessor = None
    
    # Try to load all available models
    for model_file in os.listdir(MODELS_DIR):
        if model_file.endswith('_model.pkl'):
            model_name = model_file.replace('_model.pkl', '')
            model_path = os.path.join(MODELS_DIR, model_file)
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Failed to load model {model_name}: {str(e)}")
    
    # Try to load preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            st.warning(f"Failed to load preprocessor: {str(e)}")
    
    return models, preprocessor

# Main title
st.title("🏏 IPL Score Predictor")
st.markdown("### Predict the final score of an IPL match based on the current situation")

# Check if models exist
if not os.path.exists(MODELS_DIR) or len(os.listdir(MODELS_DIR)) == 0:
    st.error("No trained models found. Please train models first.")
    
    if st.button("Generate Sample Data and Train Models"):
        st.info("This will create sample data and train models for demonstration purposes.")
        # Code to generate sample data and train models would go here
        # For now, just show a message
        st.warning("Feature not implemented yet. Please run the training scripts manually.")
    
    st.stop()

# Load models
with st.spinner("Loading models..."):
    models, preprocessor = load_models()

if not models:
    st.error("Failed to load any models. Please check the models directory.")
    st.stop()

# Create predictor
predictor = IPLScorePredictor()
predictor.preprocessor = preprocessor

# Sidebar for model selection
st.sidebar.title("Model Settings")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
predictor.model = models[selected_model_name]

# Main input form
st.subheader("Match Information")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", IPL_TEAMS)
    venue = st.selectbox("Venue", IPL_VENUES)
    current_score = st.number_input("Current Score (Runs)", min_value=0, max_value=350, value=100)
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2)
    last_five_overs_runs = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=40)

with col2:
    bowling_team = st.selectbox("Bowling Team", [team for team in IPL_TEAMS if team != batting_team])
    overs = st.slider("Overs Completed", min_value=5.0, max_value=19.5, value=12.0, step=0.1)
    striker_runs = st.number_input("Striker's Runs", min_value=0, max_value=200, value=35)
    non_striker_runs = st.number_input("Non-Striker's Runs", min_value=0, max_value=200, value=20)

# Create match features dictionary
match_features = {
    'runs': current_score,
    'wickets': wickets,
    'overs': overs,
    'striker_runs': striker_runs,
    'non_striker_runs': non_striker_runs,
    'last_five_overs_runs': last_five_overs_runs,
    'batting_team': batting_team,
    'bowling_team': bowling_team,
    'venue': venue
}

# Predict button
if st.button("Predict Final Score"):
    with st.spinner("Calculating prediction..."):
        # Calculate current run rate and projected score
        current_run_rate = current_score / overs if overs > 0 else 0
        projected_score = current_score + (current_run_rate * (20 - overs))
        
        # Get model prediction
        prediction = predictor.predict_score(match_features)
        explanation = predictor.explain_prediction(match_features)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Display prediction in a large font
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Run Rate", f"{current_run_rate:.2f}")
            st.metric("Current Score", f"{current_score}/{wickets}")
        
        with col2:
            st.metric("Projected Score (Constant RR)", f"{int(projected_score)}")
            st.metric("Overs Completed", f"{overs}/20")
        
        with col3:
            st.metric("Predicted Final Score", f"{prediction}", delta=int(prediction-projected_score))
            st.metric("Overs Remaining", f"{20-overs:.1f}")
        
        # Show feature importance if available
        if "explanation" in explanation and "feature_importance" in explanation["explanation"]:
            st.markdown("### Feature Importance")
            feature_imp = explanation["explanation"]["feature_importance"]
            
            # Convert to DataFrame for plotting
            imp_df = pd.DataFrame({
                'Feature': list(feature_imp.keys()),
                'Importance': list(feature_imp.values())
            })
            
            # Create bar chart
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         title='Feature Importance for Prediction',
                         labels={'Importance': 'Relative Importance'},
                         color='Importance')
            
            st.plotly_chart(fig)
        
        # Add some match analysis
        st.subheader("Match Analysis")
        
        if prediction > current_score * 1.5:
            st.success("The model predicts a significant acceleration in scoring rate!")
        elif prediction < projected_score:
            st.warning("The model predicts a slowdown in scoring, possibly due to wickets or conditions.")
        
        remaining_balls = (20 - overs) * 6
        runs_to_add = prediction - current_score
        
        st.info(f"The team is predicted to score {runs_to_add} more runs in the remaining {int(remaining_balls)} balls " +
                f"({runs_to_add/remaining_balls:.2f} runs per ball).")
        
        # Comparison with average scores
        avg_first_innings = 165  # This would come from historical data
        if prediction > avg_first_innings:
            st.success(f"The predicted score of {prediction} is above the average first innings score of {avg_first_innings}.")
        else:
            st.warning(f"The predicted score of {prediction} is below the average first innings score of {avg_first_innings}.")

# Add information about the model
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.info(f"This prediction is made using a {selected_model_name.replace('_', ' ').title()} model that has been " +
               "trained on historical IPL match data. The model considers the current match state, including score, " +
               "wickets, overs, batting team, bowling team, and venue to predict the final score.")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Information")
st.sidebar.markdown("""
- Data: Historical IPL match data
- Models: Linear Regression, Random Forest, XGBoost
- Features: Current score, wickets, overs, teams, venue
- Target: Final innings score
""")

# Footer
st.markdown("---")
st.markdown("#### How the Prediction Works")
st.markdown("""
The model considers:
1. **Current Match State**: Score, wickets, and overs completed
2. **Team Information**: Batting and bowling team strengths
3. **Venue**: Historical scoring patterns at the venue
4. **Batsmen Performance**: Current batsmen's scores
5. **Recent Performance**: Runs scored in the last 5 overs

Based on these factors, the model predicts the most likely final score for the innings.
""") 