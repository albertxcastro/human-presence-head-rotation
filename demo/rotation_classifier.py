from scipy.signal import find_peaks
import pandas as pd
import numpy as np

# Function to load and preprocess a csv file
def load_and_preprocess_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure it has Time and Yaw columns
    if 'Time' not in df.columns or 'Yaw' not in df.columns:
        raise ValueError(f"CSV file {file_path} must have 'Time' and 'Yaw' columns")
    
    # Extract time and yaw values
    time_values = df['Time'].values
    yaw_values = df['Yaw'].values
    
    # Create a sequence with the same format as our training data
    sequence = np.column_stack((time_values, yaw_values))
    
    # Ensure the sequence has 60 time steps by resampling if necessary
    if len(sequence) != 60:
        # Create new time points evenly spaced
        new_time_points = np.linspace(sequence[0, 0], sequence[-1, 0], 60)
        
        # Interpolate yaw values
        new_yaw_values = np.interp(new_time_points, sequence[:, 0], sequence[:, 1])
        
        # Create the resampled sequence
        sequence = np.column_stack((new_time_points, new_yaw_values))
    
    return sequence

def getFeatureVector(sequence):
    """
    Calculate engineered features from a single yaw angle sequence.
    
    Parameters:
    sequence (np.ndarray): Array of shape (sequence_length, 2) 
    where each row contains [time, yaw]
    
    Returns:
    pd.Series: Series containing the calculated features for the sequence
    """
    # Extract yaw values (index 1 of the second dimension)
    yaw_values = sequence[:, 1]

    # Find peaks and troughs
    peaks, _ = find_peaks(yaw_values)
    troughs, _ = find_peaks(-yaw_values)

    # Compute features
    num_peaks = len(peaks)
    num_troughs = len(troughs)
    
    if peaks.size > 0 and troughs.size > 0:
        amplitude_peaks = yaw_values[peaks].max() - yaw_values[troughs].min()
    else:
        amplitude_peaks = np.nan

    average_yaw = np.mean(yaw_values)
    yaw_range = np.ptp(yaw_values)  # Peak-to-peak
    std_yaw = np.std(yaw_values)

    # Return as Series
    return pd.Series({
        'num_peaks': num_peaks,
        'num_troughs': num_troughs,
        'amplitude_peaks': amplitude_peaks,
        'average_yaw': average_yaw,
        'yaw_range': yaw_range,
        'std_yaw': std_yaw
    })

def _raw_to_feats(sequences):
    """
    Applies getFeatureVector to each sequence in a batch.
    This is a helper function to make the scikit-learn pipeline work when unpickled.
    """
    feature_list = [getFeatureVector(seq) for seq in sequences]
    if not feature_list:
        return np.empty((0, 6))
    
    # Concatenate the list of Series into a DataFrame, then convert to NumPy array
    return pd.concat(feature_list, axis=1).transpose().to_numpy()
