import numpy as np
import pandas as pd
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

def robust_epoch_rejection(data, threshold=5.0, min_bad_channels=1, groups=None):
    """Flag high-amplitude epochs using channel-wise median/MAD scores.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_times)
        A single continuous time window, containing only the channels to screen.
        Non-finite values raise ValueError rather than silently passing screening.
    threshold : float
        Upper robust score cutoff (default 5). This is a starting value for visual
        review, not a validated artifact threshold or a probability.
    min_bad_channels : int
        Reject when at least this many channels exceed the cutoff in either
        temporal standard deviation or peak-to-peak amplitude.
    groups : array-like, shape (n_epochs,), optional
        Normalize separately within each group, e.g. task/event condition, so
        differences between conditions do not alone cause rejection. Each group
        needs at least five epochs; the method assumes a majority of clean epochs.

    Returns
    -------
    kept_indices, rejected_indices : ndarray of int
        Indices relative to the input, suitable for epochs.drop before any other
        deletion. No data are modified or copied into a cleaned output array.
    diagnostics : dict
        Per-epoch maximum score, bad-channel count, and per-channel scores (the
        maximum across the two amplitude metrics). Negative scores become zero.

    Notes
    -----
    Scores use (feature - median) / (MAD / 0.6744897501960817), independently
    for each channel, metric and group. One pass avoids repeatedly shrinking the
    reference distribution. If MAD is numerically zero, values above the median
    get an infinite score; equal/lower values get zero. Flat epochs/channels and
    artifacts shared by most trials require separate checks. Apply windows
    separately: concatenating them would introduce an artificial discontinuity.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3 or data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] < 2:
        raise ValueError("data must have shape (n_epochs > 0, n_channels > 0, n_times >= 2)")
    if not np.isfinite(data).all():
        raise ValueError("data contains NaN or infinity; fix these values before screening")
    if not np.isscalar(threshold) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if (isinstance(min_bad_channels, (bool, np.bool_))
            or not isinstance(min_bad_channels, (int, np.integer))
            or not 1 <= min_bad_channels <= data.shape[1]):
        raise ValueError("min_bad_channels must be an integer between 1 and n_channels")
    if groups is None:
        groups = np.zeros(len(data), dtype=int)
    groups = np.asarray(groups)
    if groups.shape != (len(data),):
        raise ValueError("groups must contain one label per epoch")

    features = np.stack([np.std(data, axis=2), np.ptp(data, axis=2)], axis=-1)
    scores = np.zeros_like(features)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 5:
            raise ValueError(f"Group {group!r} needs at least five epochs for robust screening")
        values = features[indices]
        center = np.median(values, axis=0)
        scale = np.median(np.abs(values - center), axis=0) / 0.6744897501960817
        tolerance = 32 * np.finfo(float).eps * np.maximum(
            np.max(np.abs(values), axis=0), np.finfo(float).tiny
        )
        delta = values - center
        group_scores = np.divide(delta, scale, out=np.zeros_like(values), where=scale > tolerance)
        group_scores = np.where(
            (scale <= tolerance) & (delta > tolerance), np.inf, group_scores
        )
        scores[indices] = np.maximum(group_scores, 0)

    channel_scores = scores.max(axis=2)
    bad_channel_count = (channel_scores > threshold).sum(axis=1)
    rejected = bad_channel_count >= min_bad_channels
    return np.flatnonzero(~rejected), np.flatnonzero(rejected), {
        "max_score": channel_scores.max(axis=1),
        "bad_channel_count": bad_channel_count,
        "channel_scores": channel_scores,
    }


def screen_epochs(epochs, windows=None, threshold=5.0, min_bad_channels=1, verbose=True):
    """Screen EEG epochs by condition/window without dropping or changing data.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs to screen. Event codes define the groups used by
        robust_epoch_rejection; each condition needs at least five epochs.
    windows : dict of str to (float, float), optional
        Time windows in seconds, evaluated separately. Defaults to
        {"pre": (-0.5, -0.01), "post": (0.05, 0.5)} to exclude the pulse region.
    threshold, min_bad_channels
        Forwarded to robust_epoch_rejection.
    verbose : bool
        Print per-window and total candidate counts.

    Returns
    -------
    kept_indices, rejected_indices : ndarray of int
        Current epoch positions. An epoch is rejected if any window flags it.
        Use before any other deletion, after inspecting candidates visually.
    report : pandas.DataFrame
        One row per epoch/window, containing current/original indices, event
        code, maximum score, bad-channel count and window-specific rejection.
    """
    if windows is None:
        windows = {"pre": (-0.5, -0.01), "post": (0.05, 0.5)}
    if not windows:
        raise ValueError("windows must contain at least one time window")
    for name, (tmin, tmax) in windows.items():
        if (not np.isfinite([tmin, tmax]).all() or tmin >= tmax
                or tmin < epochs.tmin or tmax > epochs.tmax):
            raise ValueError(f"Window {name!r} must be increasing and within epoch time limits")

    candidates = np.zeros(len(epochs), dtype=bool)
    tables = []
    for name, (tmin, tmax) in windows.items():
        data = epochs.get_data(picks="eeg", tmin=tmin, tmax=tmax, copy=True)
        _, rejected, diagnostics = robust_epoch_rejection(
            data, threshold=threshold, min_bad_channels=min_bad_channels,
            groups=epochs.events[:, 2],
        )
        window_mask = np.zeros(len(epochs), dtype=bool)
        window_mask[rejected] = True
        candidates |= window_mask
        tables.append(pd.DataFrame({
            "epoch_index": np.arange(len(epochs)),
            "original_index": epochs.selection.copy(),
            "event_code": epochs.events[:, 2],
            "window": name,
            "max_score": diagnostics["max_score"],
            "bad_channels": diagnostics["bad_channel_count"],
            "rejected": window_mask,
        }))
        if verbose:
            print(f"{name}: {len(rejected)}/{len(epochs)} epocas sinalizadas")

    rejected_indices = np.flatnonzero(candidates)
    if verbose:
        print(f"Total: {len(rejected_indices)}/{len(epochs)} epocas sinalizadas")
    return np.flatnonzero(~candidates), rejected_indices, pd.concat(tables, ignore_index=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from scipy.ndimage import gaussian_filter1d


def _detect_tms_peak(
    signal,
    event_sample,
    sfreq,
    search_ms=(-30, 50),
    baseline_ms=(-120, -40),
    threshold=8.0,
):
    """Detect a prominent TMS transient using the signal derivative."""

    n = len(signal)

    def bounds(window):
        return (
            event_sample + round(window[0] * sfreq / 1000),
            event_sample + round(window[1] * sfreq / 1000),
        )

    b0, b1 = bounds(baseline_ms)
    s0, s1 = bounds(search_ms)

    if min(b0, s0) < 1 or max(b1, s1) >= n - 1:
        return np.nan, np.nan

    start = min(b0, s0) - 1
    stop = max(b1, s1) + 2

    segment = signal[start:stop]

    # Temporal derivative envelope.
    envelope = np.abs(np.gradient(segment))

    sigma = max(0.5, 0.0002 * sfreq)
    envelope = gaussian_filter1d(envelope, sigma=sigma)

    baseline = envelope[b0 - start:b1 - start]
    search = envelope[s0 - start:s1 - start]

    median = np.median(baseline)
    mad = np.median(np.abs(baseline - median))
    scale = max(1.4826 * mad, np.finfo(float).eps)

    z = (search - median) / scale

    peak_idx = np.argmax(z)
    peak_score = z[peak_idx]

    if peak_score < threshold:
        return np.nan, peak_score

    peak_sample = s0 + peak_idx
    latency_ms = (
        (peak_sample - event_sample) / sfreq * 1000
    )

    return latency_ms, peak_score


def _find_latency_clusters(values, tolerance_ms=1.5, min_channels=2):
    """
    Find dense groups of latencies without assuming
    where the reference group is located.
    """
    remaining = np.asarray(values, dtype=float)
    remaining = remaining[np.isfinite(remaining)]

    clusters = []

    while len(remaining) >= min_channels:
        centers = remaining.copy()

        neighborhoods = [
            remaining[np.abs(remaining - c) <= tolerance_ms]
            for c in centers
        ]

        best = max(neighborhoods, key=len)

        if len(best) < min_channels:
            break

        center = np.median(best)

        members = remaining[
            np.abs(remaining - center) <= tolerance_ms
        ]

        clusters.append({
            "center_ms": float(np.median(members)),
            "n_channels": len(members),
        })

        remaining = remaining[
            np.abs(remaining - center) > tolerance_ms
        ]

    return sorted(
        clusters,
        key=lambda c: c["n_channels"],
        reverse=True,
    )


def synchronize_tms_eeg(
    raw,
    trigger_id,
    events=None,
    search_ms=(-30, 50),
    baseline_ms=(-120, -40),
    threshold=8.0,
    max_jitter_ms=2.0,
    group_tolerance_ms=1.5,
    min_group_channels=2,
    min_valid_events=5,
    min_valid_fraction=0.5,
    max_reference_latency_ms=10.0,
    correct=False,
    plot=True,
):
    """
    Detect temporal groups in TMS-EEG data and optionally correct
    the group shifted relative to the TMS trigger.

    Only EEG channels are analyzed and potentially shifted.
    Other channels and trigger annotations remain unchanged.
    """

    sfreq = raw.info["sfreq"]

    if events is None:
        events, event_id = mne.events_from_annotations(
            raw, verbose=False
        )

    events = np.asarray(events)
    events = events[events[:, 2] == trigger_id]

    if len(events) < min_valid_events:
        raise ValueError(
            f"Only {len(events)} TMS events found "
            f"for trigger_id={trigger_id}."
        )

    event_samples = events[:, 0] - raw.first_samp

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude="bads",
    )

    data = raw.get_data(picks=picks)

    results = []

    # --------------------------------------------------
    # 1. Detect TMS artifact latencies
    # --------------------------------------------------

    for pick, signal in zip(picks, data):

        latencies = []
        scores = []

        for sample in event_samples:

            latency, score = _detect_tms_peak(
                signal,
                sample,
                sfreq,
                search_ms=search_ms,
                baseline_ms=baseline_ms,
                threshold=threshold,
            )

            latencies.append(latency)
            scores.append(score)

        latencies = np.asarray(latencies)
        valid = latencies[np.isfinite(latencies)]

        if len(valid):
            latency = float(np.median(valid))
            jitter = float(
                np.median(np.abs(valid - latency))
            )
        else:
            latency = np.nan
            jitter = np.nan

        results.append({
            "channel": raw.ch_names[pick],
            "latency_ms": latency,
            "jitter_ms": jitter,
            "valid_events": len(valid),
            "valid_fraction": len(valid) / len(events),
            "median_score": (
                float(np.nanmedian(scores))
                if np.any(np.isfinite(scores))
                else np.nan
            ),
        })

    report = pd.DataFrame(results)

    # --------------------------------------------------
    # 2. Identify reliable channels
    # --------------------------------------------------

    reliable = (
        report["latency_ms"].notna()
        & (report["jitter_ms"] <= max_jitter_ms)
        & (report["valid_events"] >= min_valid_events)
        & (report["valid_fraction"] >= min_valid_fraction)
    )

    report["status"] = "unclassified"
    report["reliable"] = reliable

    latencies = report.loc[
        reliable, "latency_ms"
    ].to_numpy()

    # --------------------------------------------------
    # 3. Discover temporal clusters
    # --------------------------------------------------

    clusters = _find_latency_clusters(
        latencies,
        tolerance_ms=group_tolerance_ms,
        min_channels=min_group_channels,
    )

    info = {
        "n_events": len(events),
        "n_reliable_channels": int(reliable.sum()),
        "clusters": clusters,
        "delay_ms": None,
        "shift_samples": 0,
        "reference_channels": [],
        "delayed_channels": [],
        "status": "not_resolved",
    }

    # Only accept two clearly separated groups.
    if len(clusters) >= 2:

        # Avoid choosing arbitrarily when more than
        # two substantial temporal clusters exist.
        if len(clusters) == 2:

            c1, c2 = clusters

            separation = abs(
                c1["center_ms"] - c2["center_ms"]
            )

            if separation > 2 * group_tolerance_ms:

                # Select the cluster closest to trigger zero.
                ordered = sorted(
                    (c1, c2),
                    key=lambda c: abs(c["center_ms"]),
                )

                ref_center = ordered[0]["center_ms"]
                shifted_center = ordered[1]["center_ms"]

                # Do not silently assume a distant
                # artifact is synchronized to the trigger.
                if abs(ref_center) <= max_reference_latency_ms:

                    reference_mask = (
                        reliable
                        & (
                            abs(report["latency_ms"] - ref_center)
                            <= group_tolerance_ms
                        )
                    )

                    shifted_mask = (
                        reliable
                        & (
                            abs(report["latency_ms"] - shifted_center)
                            <= group_tolerance_ms
                        )
                    )

                    report.loc[
                        reference_mask, "status"
                    ] = "reference"

                    report.loc[
                        shifted_mask, "status"
                    ] = "delayed"

                    ref_latency = report.loc[
                        reference_mask, "latency_ms"
                    ].median()

                    shifted_latency = report.loc[
                        shifted_mask, "latency_ms"
                    ].median()

                    delay_ms = float(
                        shifted_latency - ref_latency
                    )

                    shift_samples = int(
                        np.round(delay_ms * sfreq / 1000)
                    )

                    info.update({
                        "delay_ms": delay_ms,
                        "shift_samples": shift_samples,
                        "reference_channels": report.loc[
                            reference_mask, "channel"
                        ].tolist(),
                        "delayed_channels": report.loc[
                            shifted_mask, "channel"
                        ].tolist(),
                        "status": "detected",
                    })

    # --------------------------------------------------
    # 4. Correct only identified shifted channels
    # --------------------------------------------------

    raw_aligned = raw.copy()

    if correct:
        if info["status"] != "detected":
            raise RuntimeError(
                "Alignment is ambiguous. No channels were changed. "
                "Inspect the diagnostic report."
            )

        shift = info["shift_samples"]

        if shift == 0:
            raise RuntimeError(
                "Estimated shift is smaller than half a sample."
            )

        raw_aligned.load_data()

        shifted_picks = [
            raw_aligned.ch_names.index(name)
            for name in info["delayed_channels"]
        ]

        k = abs(shift)

        if k >= raw_aligned.n_times:
            raise ValueError("Shift exceeds recording length.")

        for pick in shifted_picks:

            signal = raw_aligned._data[pick].copy()

            if shift > 0:
                # The shifted group is late: advance it.
                raw_aligned._data[pick, :-k] = signal[k:]
                raw_aligned._data[pick, -k:] = 0.0

            else:
                # The shifted group is early: delay it.
                raw_aligned._data[pick, k:] = signal[:-k]
                raw_aligned._data[pick, :k] = 0.0

        # The newly exposed edge contains artificial zeros.
        # MNE annotations are global, so this edge will
        # be rejected for all channels during epoching.
        edge_start = (
            (raw_aligned.n_times - k) / sfreq
            if shift > 0 else 0.0
        )

        raw_aligned.annotations.append(
            onset=edge_start,
            duration=k / sfreq,
            description="BAD_alignment_edge",
        )

        info["status"] = "corrected"

    # --------------------------------------------------
    # 5. Diagnostic visualization
    # --------------------------------------------------

    if plot:

        fig, ax = plt.subplots(figsize=(12, 5))

        colors = {
            "reference": "tab:blue",
            "delayed": "tab:orange",
            "unclassified": "gray",
        }

        for status, group in report.groupby("status"):

            ax.scatter(
                group["channel"],
                group["latency_ms"],
                color=colors[status],
                label=status,
                alpha=0.8,
            )

        ax.axhline(
            0, color="black",
            linestyle="--", alpha=0.5
        )

        ax.set_xlabel("EEG channel")
        ax.set_ylabel("TMS artifact latency (ms)")
        ax.set_title("TMS–EEG channel synchronization")
        ax.tick_params(axis="x", rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return raw_aligned, report, info