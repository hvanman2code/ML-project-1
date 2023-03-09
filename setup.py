import os
import sys
import subprocess
import platform

def install_dependencies():
    """Install the required dependencies for the IPL Score Predictor project"""
    print("Installing dependencies for IPL Score Predictor...")
    
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, 'requirements.txt')
    
    if not os.path.exists(requirements_path):
        print(f"Error: Requirements file not found at {requirements_path}")
        return False
    
    # Install dependencies
    try:
        print(f"Installing packages from {requirements_path}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_path], check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_virtual_environment():
    """Create a virtual environment for the project"""
    print("Creating virtual environment...")
    
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(current_dir, 'venv')
    
    if os.path.exists(venv_dir):
        print(f"Virtual environment already exists at {venv_dir}")
        return venv_dir
    
    # Create virtual environment
    try:
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
        print(f"Virtual environment created at {venv_dir}")
        return venv_dir
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return None

def setup_project():
    """Setup the IPL Score Predictor project"""
    print("=== IPL Score Predictor Setup ===\n")
    
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Create virtual environment (optional)
    create_venv = input("Do you want to create a virtual environment? (y/n): ").lower() == 'y'
    venv_dir = None
    
    if create_venv:
        venv_dir = create_virtual_environment()
        if not venv_dir:
            print("Failed to create virtual environment. Proceeding with system Python...")
    
    # Step 2: Install dependencies
    if install_dependencies():
        print("\nSetup completed successfully!")
        
        # Step 3: Provide instructions
        print("\n=== Getting Started ===")
        if venv_dir:
            if platform.system() == 'Windows':
                activate_cmd = f"{venv_dir}\\Scripts\\activate"
            else:
                activate_cmd = f"source {venv_dir}/bin/activate"
            print(f"1. Activate the virtual environment: {activate_cmd}")
        
        print("2. Generate sample data and train models:")
        print(f"   {sys.executable} {os.path.join(current_dir, 'src', 'run_pipeline.py')}")
        
        print("3. Run the web application:")
        print(f"   {sys.executable} {os.path.join(current_dir, 'run_app.py')}")
        
        print("\nEnjoy predicting IPL scores!")
    else:
        print("\nSetup failed. Please check the error messages and try again.")

if __name__ == "__main__":
    setup_project() 