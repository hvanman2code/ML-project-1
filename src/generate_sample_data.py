import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Define constants
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

def generate_sample_data(num_samples=1000, seed=42):
    """
    Generate synthetic IPL match data for model training and testing
    
    Args:
        num_samples: Number of sample records to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic IPL match data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    data = []
    
    start_date = datetime(2023, 3, 31)  # IPL 2023 start date
    
    for i in range(num_samples):
        # Generate match date
        match_date = start_date + timedelta(days=random.randint(0, 60))
        
        # Select teams randomly ensuring they are different
        teams = random.sample(IPL_TEAMS, 2)
        batting_team, bowling_team = teams
        
        # Select venue randomly
        venue = random.choice(IPL_VENUES)
        
        # Generate match state (current score, wickets, overs)
        overs = round(random.uniform(5.0, 19.5), 1)  # Between 5 and 19.5 overs
        wickets = random.randint(0, min(9, int(overs/2)))  # More overs, higher chance of wickets
        
        # Base run rate varies by team and venue
        base_rr = random.uniform(7.0, 9.5)
        
        # Teams have different scoring patterns
        if batting_team in ["Mumbai Indians", "Royal Challengers Bangalore", "Punjab Kings"]:
            base_rr += 0.5  # Aggressive teams
        elif batting_team in ["Chennai Super Kings", "Kolkata Knight Riders"]:
            base_rr += 0.3  # Balanced teams
            
        # Venues have different scoring patterns
        if venue in ["M. Chinnaswamy Stadium", "Wankhede Stadium"]:
            base_rr += 0.7  # High-scoring venues
        elif venue in ["M. A. Chidambaram Stadium", "Arun Jaitley Stadium"]:
            base_rr -= 0.3  # Lower-scoring venues
            
        # Wickets affect run rate
        wicket_factor = max(0.7, 1 - (wickets * 0.05))
        
        # Calculate current run rate with some noise
        current_rr = base_rr * wicket_factor * random.uniform(0.9, 1.1)
        
        # Calculate current score
        runs = int(current_rr * overs)
        
        # Generate batsmen scores
        total_contribution = random.uniform(0.7, 0.9)  # How much of total score is from top batsmen
        striker_ratio = random.uniform(0.4, 0.7)  # How the score is split between striker and non-striker
        
        striker_runs = int(runs * total_contribution * striker_ratio)
        non_striker_runs = int(runs * total_contribution * (1 - striker_ratio))
        
        # Ensure scores make sense
        striker_runs = min(striker_runs, runs - 10)
        non_striker_runs = min(non_striker_runs, runs - striker_runs)
        
        # Generate runs in last five overs
        last_five_overs = min(5, overs)
        last_five_overs_runs = int(last_five_overs * current_rr * random.uniform(0.8, 1.3))
        last_five_overs_runs = min(last_five_overs_runs, runs)
        
        # Calculate final score (target variable)
        # Final run rate tends to change in the last few overs
        remaining_overs = 20 - overs
        
        if remaining_overs > 0:
            # Factors affecting final run rate:
            # 1. Wickets in hand
            wickets_in_hand = 10 - wickets
            wicket_multiplier = 0.7 + (0.3 * (wickets_in_hand / 10))
            
            # 2. Current run rate momentum
            if last_five_overs_runs / last_five_overs > current_rr:
                momentum = random.uniform(1.0, 1.2)  # Accelerating
            else:
                momentum = random.uniform(0.8, 1.0)  # Slowing down
                
            # 3. Phase of the innings
            if overs < 10:
                phase_factor = random.uniform(1.1, 1.3)  # Middle overs acceleration
            elif overs < 15:
                phase_factor = random.uniform(1.2, 1.5)  # Building for death overs
            else:
                phase_factor = random.uniform(1.3, 1.7)  # Death overs acceleration
                
            # Calculate projected final run rate and add some noise
            final_rr = current_rr * wicket_multiplier * momentum * phase_factor * random.uniform(0.9, 1.1)
            
            # Calculate additional runs
            additional_runs = int(final_rr * remaining_overs)
            
            # Calculate final score
            final_score = runs + additional_runs
        else:
            final_score = runs
        
        # Create data point
        data_point = {
            'match_date': match_date.strftime('%Y-%m-%d'),
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'runs': runs,
            'wickets': wickets,
            'overs': overs,
            'striker_runs': striker_runs,
            'non_striker_runs': non_striker_runs,
            'last_five_overs_runs': last_five_overs_runs,
            'final_score': final_score
        }
        
        data.append(data_point)
    
    return pd.DataFrame(data)

def save_data(data, file_path):
    """
    Save the generated data to a CSV file
    
    Args:
        data: DataFrame to save
        file_path: Path to save the CSV file
    """
    data.to_csv(file_path, index=False)
    print(f"Sample data saved to {file_path}")
    
def train_test_split(data, test_size=0.2, seed=42):
    """
    Split the data into training and testing sets
    
    Args:
        data: DataFrame to split
        test_size: Proportion of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_data, test_data
    """
    np.random.seed(seed)
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split index
    split_idx = int(len(data) * (1 - test_size))
    
    # Split the data
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate sample data
    print("Generating sample IPL match data...")
    sample_data = generate_sample_data(num_samples=1500)
    
    # Split into training and testing sets
    train_data, test_data = train_test_split(sample_data)
    
    # Save the data
    save_data(sample_data, os.path.join(data_dir, 'ipl_sample_data.csv'))
    save_data(train_data, os.path.join(data_dir, 'ipl_train_data.csv'))
    save_data(test_data, os.path.join(data_dir, 'ipl_test_data.csv'))
    
    # Copy to raw data for processing pipeline
    save_data(sample_data, os.path.join(data_dir, 'ipl_raw_data.csv'))
    
    print("Sample data generation complete.")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Display sample records
    print("\nSample records:")
    print(sample_data.head()) 