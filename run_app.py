import os
import sys
import subprocess

def run_streamlit_app():
    """Run the Streamlit web application"""
    print("Starting IPL Score Predictor Web Application...")
    
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, 'src', 'app.py')
    
    if not os.path.exists(app_path):
        print(f"Error: App file not found at {app_path}")
        return
    
    # Check if models exist
    models_dir = os.path.join(current_dir, 'models')
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        print("Warning: No trained models found in the models directory.")
        print("Running the data generation and model training pipeline first...")
        
        # Run the pipeline to generate data and train models
        pipeline_path = os.path.join(current_dir, 'src', 'run_pipeline.py')
        try:
            subprocess.run([sys.executable, pipeline_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline: {e}")
            return
    
    # Run the Streamlit app
    try:
        print("\nStarting Streamlit server...")
        print("Once the server is running, you can access the app at http://localhost:8501")
        subprocess.run(['streamlit', 'run', app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
    except FileNotFoundError:
        print("Error: Streamlit not found. Please make sure it's installed.")
        print("You can install it using: pip install streamlit")

if __name__ == "__main__":
    run_streamlit_app() 