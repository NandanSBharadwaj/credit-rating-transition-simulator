# Complete S&P Rating Transition Matrix Analysis
# Credit Risk Modeling and Simulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from google.colab import files

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("=== S&P RATING TRANSITION MATRIX ANALYSIS ===\n")

# Upload the CSV file
print("Please select your S&P rating transition matrix CSV file:")
uploaded = files.upload()

# Get the uploaded filename
filename = list(uploaded.keys())[0]
print(f"\nUploaded file: {filename}")

# Load and clean the matrix
print("\nLoading and cleaning the transition matrix...")

# Check file extension and handle different formats
file_extension = filename.lower().split('.')[-1]
print(f"File extension: {file_extension}")

try:
    if file_extension == 'csv':
        matrix = pd.read_csv(filename, index_col=0)
    elif file_extension in ['numbers', 'sffnumbers']:
        print("Error: Apple Numbers format detected!")
        print("Please convert your file to CSV format:")
        print("1. Open the file in Apple Numbers")
        print("2. Go to File → Export To → CSV")
        print("3. Save as a .csv file")
        print("4. Upload the CSV file instead")
        raise ValueError("Numbers files are not supported. Please convert to CSV.")
    elif file_extension in ['xlsx', 'xls']:
        matrix = pd.read_excel(filename, index_col=0)
    else:
        # Try different encodings for CSV
        for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
            try:
                print(f"Trying encoding: {encoding}")
                matrix = pd.read_csv(filename, index_col=0, encoding=encoding)
                print(f"Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read file with any supported encoding")
            
except Exception as e:
    print(f"Error reading file: {e}")
    print("\nPlease ensure your file is in one of these formats:")
    print("- CSV (.csv)")
    print("- Excel (.xlsx, .xls)")
    print("\nIf you have a Numbers file, please convert it to CSV first.")
    raise

# Remove completely empty rows and columns
matrix = matrix.dropna(how='all', axis=0)  # Remove rows where ALL values are NaN
matrix = matrix.dropna(how='all', axis=1)  # Remove columns where ALL values are NaN

print(f"Initial matrix shape: {matrix.shape}")
print("Initial matrix:")
print(matrix)

# Convert from percentage to decimal
matrix = matrix / 100

# ============================================================================
# 2. MATRIX DIAGNOSTICS AND CLEANING
# ============================================================================

print("\n=== MATRIX DIAGNOSTICS ===")
print(f"Shape: {matrix.shape}")
print(f"Contains NaN: {matrix.isnull().any().any()}")
print(f"Contains negative values: {(matrix < 0).any().any()}")

# Check row sums
row_sums = matrix.sum(axis=1)
print(f"\nInitial row sums:")
print(row_sums.round(4))

# Clean the matrix thoroughly
print("\nCleaning matrix...")
matrix_clean = matrix.copy()

# Replace NaN with 0
matrix_clean = matrix_clean.fillna(0)

# Make sure no negative values
matrix_clean = matrix_clean.clip(lower=0)

# Normalize each row to sum to 1.0
row_sums = matrix_clean.sum(axis=1)
matrix_clean = matrix_clean.div(row_sums, axis=0)

print("Matrix cleaned!")
print("New row sums:", matrix_clean.sum(axis=1).round(4))
print("Contains NaN:", matrix_clean.isnull().any().any())

# ============================================================================
# 3. MATRIX ORDERING AND SETUP
# ============================================================================

# Define the correct order for credit ratings (best to worst)
correct_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

# Filter to only include ratings that exist in our matrix
available_ratings = [rating for rating in correct_order if rating in matrix_clean.index]
print(f"\nAvailable ratings: {available_ratings}")

# Reorder both rows and columns to match the standard rating order
matrix_final = matrix_clean.reindex(index=available_ratings, columns=available_ratings).fillna(0)

# Re-normalize after reordering (in case any NaN values were filled)
row_sums = matrix_final.sum(axis=1)
matrix_final = matrix_final.div(row_sums, axis=0)

print("\nProperly ordered matrix:")
print(matrix_final)
print(f"Final matrix shape: {matrix_final.shape}")

# Set up simulation variables using the clean, ordered matrix
states = matrix_final.index.tolist()
n_states = len(states)
transition_matrix = matrix_final.values

# Create mapping dictionaries
state_to_index = {state: i for i, state in enumerate(states)}
index_to_state = {i: state for i, state in enumerate(states)}

print(f"\nStates in correct order: {states}")
print(f"Number of states: {n_states}")
print(f"Final row sums: {matrix_final.sum(axis=1).round(4)}")

# ============================================================================
# 4. SIMULATION FUNCTIONS
# ============================================================================

def simulate_path(start_rating, n_years):
    """Simulate a single rating path over n_years starting from start_rating"""
    path = [start_rating]
    current_state = state_to_index[start_rating]
    
    for year in range(n_years):
        probs = transition_matrix[current_state]
        next_state = np.random.choice(n_states, p=probs)
        path.append(index_to_state[next_state])
        current_state = next_state
    
    return path

def run_simulations(start_rating, n_years, n_simulations):
    """Run multiple simulations and calculate default rate"""
    defaults = 0
    all_paths = []
    
    for sim in range(n_simulations):
        path = simulate_path(start_rating, n_years)
        all_paths.append(path)
        
        if 'D' in path:  # Check if default occurred
            defaults += 1
    
    default_rate = defaults / n_simulations
    print(f"Default rate over {n_years} years: {default_rate:.2%}")
    
    return all_paths, default_rate

# ============================================================================
# 5. TEST SIMULATIONS
# ============================================================================

print("\n=== TESTING SIMULATIONS ===")

# Test individual path simulation
print("\nTesting individual path simulation:")
test_path = simulate_path('BBB', 5)
print(f"5-year path starting from BBB: {test_path}")

# Test multiple simulations
print(f"\nRunning 1000 simulations for BBB rating over 5 years...")
paths, default_rate = run_simulations('BBB', 5, 1000)
print(f"Generated {len(paths)} paths")
print(f"Sample path: {paths[0]}")

# ============================================================================
# 6. BOND VALUATION ANALYSIS
# ============================================================================

print("\n=== BOND VALUATION ANALYSIS ===")

# Define rating-to-price mapping (bond values)
rating_to_price = {
    "AAA": 100,
    "AA": 98,
    "A": 95,
    "BBB": 90,
    "BB": 80,
    "B": 70,
    "CCC": 50,
    "D": 0
}

def calculate_value_path(path, rating_to_price):
    """Convert rating path to value path"""
    return [rating_to_price.get(rating, 0) for rating in path]

# Calculate value paths for all simulations
all_value_paths = [calculate_value_path(path, rating_to_price) for path in paths]
avg_value_per_year = np.mean(all_value_paths, axis=0)

print("Average bond value per year:", [f"{val:.2f}" for val in avg_value_per_year])

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("\n=== CREATING VISUALIZATIONS ===")

# Plot 1: Final Rating Distribution
final_ratings = [path[-1] for path in paths]
counts = Counter(final_ratings)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(counts.keys(), counts.values())
plt.title("Final Rating Distribution After 5 Years")
plt.xlabel("Credit Rating")
plt.ylabel("Number of Bonds")
plt.xticks(rotation=45)

# Plot 2: Average Bond Value Over Time
plt.subplot(1, 2, 2)
plt.plot(range(len(avg_value_per_year)), avg_value_per_year, marker='o', linewidth=2)
plt.title("Average Bond Value Over 5 Years")
plt.xlabel("Year")
plt.ylabel("Bond Value")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 8. RECESSION SCENARIO ANALYSIS
# ============================================================================

print("\n=== RECESSION SCENARIO ANALYSIS ===")

# Create recession scenario matrix
recession_matrix = matrix_final.copy()

# Increase downgrade and default probabilities for BBB (example scenario)
if 'BBB' in recession_matrix.index:
    print("Creating recession scenario for BBB rating...")
    
    # Increase probability of downgrades and default
    if 'BB' in recession_matrix.columns:
        recession_matrix.loc["BBB", "BB"] += 0.05  # +5% chance of downgrade to BB
    if 'D' in recession_matrix.columns:
        recession_matrix.loc["BBB", "D"] += 0.02   # +2% chance of default
    
    # Decrease probability of staying at BBB
    recession_matrix.loc["BBB", "BBB"] -= 0.07   # -7% to balance
    
    # Re-normalize the BBB row
    row = recession_matrix.loc["BBB"]
    recession_matrix.loc["BBB"] = row / row.sum()
    
    print("Recession matrix BBB row:", recession_matrix.loc["BBB"].round(4))
    
    # Update transition matrix for recession scenario
    transition_matrix_recession = recession_matrix.values
    original_transition_matrix = transition_matrix.copy()
    transition_matrix = transition_matrix_recession
    
    # Run recession scenario simulation
    print("\nRunning recession scenario simulation...")
    recession_paths, recession_default_rate = run_simulations('BBB', 5, 1000)
    
    # Restore original matrix
    transition_matrix = original_transition_matrix

# ============================================================================
# 9. EXPECTED LOSS CALCULATION
# ============================================================================

print("\n=== EXPECTED LOSS CALCULATION ===")

# Define Loss Given Default (LGD) - typically 40-60% for corporate bonds
LGD = 0.45  # 45% loss given default

# Calculate expected loss for normal scenario
normal_expected_loss = default_rate * LGD
print(f"Normal scenario - Default rate: {default_rate:.2%}")
print(f"Normal scenario - Expected loss: {normal_expected_loss:.2%}")

# Calculate expected loss for recession scenario (if applicable)
if 'recession_default_rate' in locals():
    recession_expected_loss = recession_default_rate * LGD
    print(f"Recession scenario - Default rate: {recession_default_rate:.2%}")
    print(f"Recession scenario - Expected loss: {recession_expected_loss:.2%}")
    print(f"Additional risk in recession: {(recession_expected_loss - normal_expected_loss):.2%}")

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================

print("\n=== SUMMARY STATISTICS ===")
print(f"Transition Matrix Dimensions: {matrix_final.shape}")
print(f"Credit Rating States: {states}")
print(f"Simulation Parameters: BBB rating, 5 years, 1000 paths")
print(f"Average Starting Bond Value: {rating_to_price.get('BBB', 'N/A')}")
print(f"Average Ending Bond Value: {avg_value_per_year[-1]:.2f}")
print(f"Total Value Decline: {rating_to_price.get('BBB', 90) - avg_value_per_year[-1]:.2f}")
print(f"Loss Given Default (LGD): {LGD:.1%}")

print(f"\n=== ANALYSIS COMPLETE ===")
print("The transition matrix is now ready for further credit risk analysis!")
print("Variables available for use:")
print("- matrix_final: Clean, normalized transition matrix")
print("- transition_matrix: NumPy array for simulations") 
print("- paths: List of all simulated rating paths")
print("- avg_value_per_year: Average bond values over time")