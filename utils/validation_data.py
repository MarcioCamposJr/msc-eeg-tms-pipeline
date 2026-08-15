import numpy as np
import mne
from statsmodels.tsa.stattools import adfuller

def filter_stationary_epochs(epochs, p_value_threshold=0.05, channel_threshold=0.95):
    """
    Tests the stationarity of each trial (epoch) in MNE Epochs using the ADF test.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The MNE Epochs object containing the data.
    p_value_threshold : float
        P-value threshold for the Augmented Dickey-Fuller (ADF) test. 
        If p_value < threshold, the time series is considered stationary (default: 0.05).
    channel_threshold : float
        Minimum proportion (from 0 to 1) of channels that must be stationary 
        for the epoch to be kept. The default 0.95 requires at least 95% 
        of the channels in the current epoch to be stationary.
        
    Returns:
    --------
    stationary_epochs : mne.Epochs
        A new MNE Epochs object containing only the epochs that passed the test.
    """
    # Extract data as a NumPy array with shape (n_epochs, n_channels, n_times).
    # We select only the 'eeg' channels to avoid testing reference/stim channels.
    data = epochs.get_data(picks='eeg') 
    n_epochs, n_channels, n_times = data.shape
    
    epochs_to_keep = []
    
    print("Testing stationarity (ADF Test)... This might take a few seconds depending on data size.")
    
    for i in range(n_epochs):
        stationary_channels_count = 0
        
        for j in range(n_channels):
            # Get the time series for a single channel (j) in the current epoch (i)
            time_series = data[i, j, :]
            
            try:
                # Run the ADF test. The second return value [1] is the p-value.
                # adfuller can fail if the series has zero variance (e.g., constant line)
                adf_result = adfuller(time_series)
                p_value = adf_result[1]
                
                # If p-value < threshold, we reject the null hypothesis of non-stationarity
                if p_value < p_value_threshold:
                    stationary_channels_count += 1
                    
            except ValueError:
                # If the test raises an error (e.g. zero variance), we consider the channel non-stationary
                pass
                
        # Calculate the proportion of stationary channels for the current epoch
        prop_stationary = stationary_channels_count / n_channels
        
        # If the proportion meets our threshold, record the epoch index to keep it
        if prop_stationary >= channel_threshold:
            epochs_to_keep.append(i)
            
    # Summary of the operation
    print("\n--- Summary ---")
    print(f"Total epochs evaluated: {n_epochs}")
    print(f"Stationary epochs KEPT: {len(epochs_to_keep)}")
    print(f"Non-stationary epochs DROPPED: {n_epochs - len(epochs_to_keep)}")
    
    # MNE allows slicing an Epochs object by passing a list of indices.
    # This returns a new Epochs object containing only the trials of interest.
    stationary_epochs = epochs[epochs_to_keep]
    
    return stationary_epochs

def check_hand(state):
    if 'right' in state:
        return 'Right hand'
    elif 'left' in state:
        return 'Left hand'
    elif 'bilateral' in state:
        return 'Bilateral hand'