import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import mne

def interpolate_channels(inst, channels, montage="standard_1020", copy=True):
    """Spatially interpolate selected EEG channels, recreating dropped channels.

    Parameters
    ----------
    inst : Raw | Epochs | Evoked
        Preloaded EEG data. Non-EEG channels are preserved.
    channels : list of str
        EEG channels to reconstruct, whether present or previously dropped.
    montage : str | DigMontage | None
        Electrode positions, defaulting to the standard_1020 montage used in
        the TEP notebook. Pass the measured montage for measured positions.
        None keeps the current montage (all requested positions must exist).
    copy : bool
        Return a copy by default. False modifies the input in place.

    Returns
    -------
    inst : Raw | Epochs | Evoked
        Data with requested channels interpolated using spherical splines.
        Other bad channels remain marked and are excluded as donors.
        Recreated channels are appended. Reference and baseline are unchanged.
    """
    if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        raise ValueError("inst must be an instance of Raw, Epochs, or Evoked")
    if isinstance(channels, str):
        channels = [channels]
    channels = list(dict.fromkeys(channels))
    result = inst.copy() if copy else inst
    if not channels:
        return result

    existing = [ch for ch in channels if ch in result.ch_names]
    if existing and any(kind != "eeg" for kind in result.get_channel_types(picks=existing)):
        raise ValueError("Only EEG channels can be interpolated")
    missing = [ch for ch in channels if ch not in result.ch_names]
    target_montage = (
        mne.channels.make_standard_montage(montage)
        if isinstance(montage, str) else montage
    )
    positions = target_montage if target_montage is not None else result.get_montage()
    if positions is None or any(ch not in positions.ch_names for ch in channels):
        raise ValueError("Provide a montage containing positions for every requested channel")

    original_bads = list(result.info["bads"])
    if not isinstance(result, mne.Evoked):
        result.load_data()
    if missing:
        # Add zero-filled EEG placeholders; interpolation replaces their data.
        mne.add_reference_channels(result, missing, copy=False)
    result.set_montage(positions)
    other_bads = [ch for ch in original_bads if ch not in channels]
    result.info["bads"] = list(dict.fromkeys(original_bads + channels))
    result.interpolate_bads(
        method=dict(eeg="spline"), reset_bads=False, exclude=other_bads,
    )
    result.info["bads"] = other_bads
    return result


def fix_stim_artifact_cubic(inst, events=None, event_id=None, tmin=0.0, tmax=0.01, pre_window=0.010, post_window=0.010, method="cubic"):
    """
    Interpolate stimulus artifact using cubic spline or PCHIP interpolation.
    
    Parameters:
    inst : instance of Raw, Epochs, or Evoked
        The data to be processed.
    events : array, shape (n_events, 3)
        The events array. Required if inst is Raw.
    event_id : int | list of int
        The event ID(s) to interpolate around. Required if inst is Raw.
    tmin : float
        Start time of the interpolation window around the event (in seconds).
    tmax : float
        End time of the interpolation window around the event (in seconds).
    pre_window : float
        Duration of the time window before tmin used for fitting the cubic spline (in seconds).
    post_window : float
        Duration of the time window after tmax used for fitting the cubic spline (in seconds).
    method : {"cubic", "pchip"}
        Interpolator to use. "cubic" preserves the original CubicSpline behavior;
        "pchip" uses shape-preserving piecewise cubic interpolation. Both use
        only the pre/post support windows, excluding the replaced samples.
        
    Returns:
    inst : instance of Raw, Epochs, or Evoked
        The modified data (modified in-place).
    """
    if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        raise ValueError('inst must be an instance of Raw, Epochs, or Evoked')

    if method not in ("cubic", "pchip"):
        raise ValueError('method must be "cubic" or "pchip"')
    interpolator = CubicSpline if method == "cubic" else PchipInterpolator
        
    sfreq = inst.info['sfreq']
    
    # Convert times to samples
    smin_interp = int(round(tmin * sfreq))
    smax_interp = int(round(tmax * sfreq))
    s_pre = int(round(pre_window * sfreq))
    s_post = int(round(post_window * sfreq))
    
    if s_pre <= 0 or s_post <= 0:
        raise ValueError("pre_window and post_window must be > 0")
        
    if isinstance(inst, mne.io.BaseRaw):
        if events is None or event_id is None:
            raise ValueError('events and event_id must be provided for Raw data')
        
        if isinstance(event_id, int):
            event_id = [event_id]
            
        event_samps = events[np.isin(events[:, 2], event_id), 0]
        data = inst._data
        first_samp = inst.first_samp
        
        for samp in event_samps:
            samp_rel = samp - first_samp
            
            idx_interp_start = samp_rel + smin_interp
            idx_interp_end = samp_rel + smax_interp
            
            idx_fit_pre_start = idx_interp_start - s_pre
            idx_fit_pre_end = idx_interp_start
            
            idx_fit_post_start = idx_interp_end
            idx_fit_post_end = idx_interp_end + s_post
            
            if idx_fit_pre_start < 0 or idx_fit_post_end > data.shape[1]:
                continue
                
            x_fit = np.concatenate([
                np.arange(idx_fit_pre_start, idx_fit_pre_end),
                np.arange(idx_fit_post_start, idx_fit_post_end)
            ])
            x_interp = np.arange(idx_interp_start, idx_interp_end)
            
            if len(x_interp) == 0:
                continue
                
            # Fit and interpolate across all channels at once
            y_fit = data[:, x_fit]
            cs = interpolator(x_fit, y_fit, axis=1)
            data[:, x_interp] = cs(x_interp)
            
    else:
        # For Epochs or Evoked
        idx_0 = inst.time_as_index(0.0)[0]
        idx_interp_start = idx_0 + smin_interp
        idx_interp_end = idx_0 + smax_interp
        
        idx_fit_pre_start = idx_interp_start - s_pre
        idx_fit_pre_end = idx_interp_start
        
        idx_fit_post_start = idx_interp_end
        idx_fit_post_end = idx_interp_end + s_post
        
        data = inst._data
        n_dim = data.ndim
        
        if idx_fit_pre_start < 0 or idx_fit_post_end > data.shape[-1]:
            raise ValueError("Time windows go beyond the data limits.")
            
        x_fit = np.concatenate([
            np.arange(idx_fit_pre_start, idx_fit_pre_end),
            np.arange(idx_fit_post_start, idx_fit_post_end)
        ])
        x_interp = np.arange(idx_interp_start, idx_interp_end)
        
        if len(x_interp) > 0:
            if n_dim == 2:  # Evoked
                y_fit = data[:, x_fit]
                cs = interpolator(x_fit, y_fit, axis=1)
                data[:, x_interp] = cs(x_interp)
            elif n_dim == 3:  # Epochs
                y_fit = data[:, :, x_fit]
                cs = interpolator(x_fit, y_fit, axis=2)
                data[:, :, x_interp] = cs(x_interp)
                
    return inst

from scipy import signal

def custom_lowpass_butterworth(epochs, h_freq=80.0, order=4):
    """
    Aplica um filtro passa-baixa IIR (Butterworth) com distorção de fase zero 
    em um objeto MNE Epochs.
    """
    # Cria uma cópia para preservar os dados originais
    epochs_filtered = epochs.copy()
    
    # Extrai a frequência de amostragem
    sfreq = epochs_filtered.info['sfreq']
    
    # Calcula a frequência de Nyquist
    nyq = sfreq / 2.0
    
    # Normaliza a frequência de corte
    wn = h_freq / nyq
    
    # Cria os coeficientes do filtro Butterworth (b, a)
    b, a = signal.butter(N=order, Wn=wn, btype='low')
    
    # Função wrapper para o SciPy filtfilt
    def apply_filtfilt(x):
        # filtfilt roda o filtro forward e backward para zerar o atraso de fase
        return signal.filtfilt(b, a, x)
    
    # Aplica a função em todos os dados do objeto Epochs
    epochs_filtered.apply_function(apply_filtfilt, picks='all')
    
    return epochs_filtered
