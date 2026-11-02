import numpy as np
import mne
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed

def _is_stationary(time_series, p_value_threshold):
    """
    Função auxiliar para rodar no worker paralelo.
    Testa a estacionariedade de uma única série temporal.
    """
    try:
        # adfuller retorna uma tupla onde o índice [1] é o p-valor
        p_value = adfuller(time_series)[1]
        return p_value < p_value_threshold
    except ValueError:
        # Lida com casos de variância zero (sinal constante/flat)
        return False

def filter_stationary_epochs(epochs, p_value_threshold=0.05, channel_threshold=0.95, n_jobs=-1):
    """
    Tests the stationarity of each trial (epoch) in MNE Epochs using the ADF test in parallel.

    Parameters:
    -----------
    epochs : mne.Epochs
        The MNE Epochs object containing the data.
    p_value_threshold : float
        P-value threshold for the Augmented Dickey-Fuller (ADF) test.
    channel_threshold : float
        Minimum proportion (from 0 to 1) of channels that must be stationary
        for the epoch to be kept.
    n_jobs : int
        Number of CPU cores to use. -1 uses all available cores (default: -1).

    Returns:
    --------
    stationary_epochs : mne.Epochs
        A new MNE Epochs object containing only the epochs that passed the test.
    """
    # Extrai os dados: shape (n_epochs, n_channels, n_times)
    data = epochs.get_data(picks='eeg')
    n_epochs, n_channels, n_times = data.shape

    print(f"Testing stationarity (ADF Test) on {n_epochs * n_channels} signals using {n_jobs if n_jobs != -1 else 'all'} CPU cores...")

    # ACHATAMENTO (FLATTEN):
    # Transforma o array 3D em 2D para eliminar os loops aninhados.
    # O reshape cria apenas uma "view" (visão), não duplicando os dados na memória RAM.
    flat_data = data.reshape(-1, n_times)

    # PARALELISMO:
    # Dispara o cálculo em múltiplos núcleos simultaneamente.
    # Retorna uma lista 1D de booleanos (True/False)
    is_stationary_flat = Parallel(n_jobs=n_jobs)(
        delayed(_is_stationary)(ts, p_value_threshold) for ts in flat_data
    )

    # VOLTANDO AO FORMATO ORIGINAL:
    # Transforma a lista de resultados (1D) de volta para o formato (n_epochs, n_channels)
    is_stationary_matrix = np.array(is_stationary_flat).reshape(n_epochs, n_channels)

    # VETORIZAÇÃO COM NUMPY:
    # No Python, True = 1 e False = 0.
    # Portanto, a média no eixo 1 (canais) dá exatamente a proporção de canais estacionários por epoch!
    prop_stationary = is_stationary_matrix.mean(axis=1)

    # Encontra os índices das epochs que atingiram o limite mínimo (channel_threshold)
    epochs_to_keep = np.where(prop_stationary >= channel_threshold)[0]

    # Summary of the operation
    print("\n--- Summary ---")
    print(f"Total epochs evaluated: {n_epochs}")
    print(f"Stationary epochs KEPT: {len(epochs_to_keep)}")
    print(f"Non-stationary epochs DROPPED: {n_epochs - len(epochs_to_keep)}")

    # Filtra e retorna apenas as epochs desejadas
    stationary_epochs = epochs[epochs_to_keep]

    return stationary_epochs

def check_hand(state):
    if 'right' in state:
        return 'Right hand'
    elif 'left' in state:
        return 'Left hand'
    elif 'bilateral' in state:
        return 'Bilateral hand'


def iterative_zscore_rejection(data, threshold=2.5):
    """
    Iteratively reject epochs based on their standard deviation.

    Parameters
    ----------
    data : ndarray
        Shape (n_epochs, n_channels, n_times).
    threshold : float
        Z-score threshold for rejection.

    Returns
    -------
    clean_data : ndarray
        Data after rejection.
    kept_indices : ndarray
        Original indices of retained epochs.
    rejected_indices : ndarray
        Original indices of rejected epochs.
    """
    epoch_std = np.std(data, axis=(1, 2))

    kept_indices = np.arange(len(data))
    rejected_indices = []

    while len(kept_indices) > 1:
        current_std = epoch_std[kept_indices]

        mean = np.mean(current_std)
        std = np.std(current_std)

        if std == 0:
            break

        z_scores = (current_std - mean) / std

        # Reject only high-variance epochs
        outliers = z_scores > threshold

        if not np.any(outliers):
            break

        rejected_indices.extend(kept_indices[outliers])
        kept_indices = kept_indices[~outliers]

    return (
        data[kept_indices],
        kept_indices,
        np.array(rejected_indices, dtype=int),
    )
