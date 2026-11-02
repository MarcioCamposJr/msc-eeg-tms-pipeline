# ------------------- NB! -----------------------------------
# This is a beta version of the online Sound algorithm and is not yet fully validated.
# The given lead field matrix LFM_Aalto_ReftepPP.csv only works with the Aalto Reftep++ 64-ch channel configuration.
# If your channel configuration is different, please make your own lead field.
# If you are planning to use this code for experiments, please contact Tuomas Mutanen and Matilda Makkonen.
# The optimal parameters might be different for your use case.
# ------------------- NB! -----------------------------------

# Authors: Tuomas Mutanen, Olli-Pekka Kahilakoski, Matilda Makkonen, Johanna Metsomaa
# Modified by Oskari Ahola and Lisa Haxel

import time

import numpy as np

def sound(eeg_samples, baseline_correction, sigmas, num_of_channels, lfm, iterations, lambda0, convergence_boundary, fixed_iters=False):
    """Return the spatial filter, noise estimates, iterations and convergence."""
    eeg_samples = np.asarray(eeg_samples, dtype=float)
    lfm = np.asarray(lfm, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    if (eeg_samples.ndim != 2 or eeg_samples.shape[0] < 2
            or num_of_channels < 2 or eeg_samples.shape[1] != num_of_channels
            or lfm.ndim != 2 or lfm.shape[0] != num_of_channels or lfm.shape[1] == 0
            or sigmas.shape != (num_of_channels,)):
        raise ValueError("SOUND requires samples x channels and a matching channels x sources leadfield.")
    if not all(np.isfinite(x).all() for x in (eeg_samples, lfm, sigmas, baseline_correction)):
        raise ValueError("SOUND inputs must be finite.")
    if (not isinstance(iterations, (int, np.integer)) or iterations < 1
            or not np.isfinite(lambda0) or lambda0 <= 0
            or not np.isfinite(convergence_boundary) or convergence_boundary < 0
            or np.any(sigmas <= 0) or not np.any(lfm)):
        raise ValueError("SOUND requires positive iterations, regularization, noise estimates and a nonzero leadfield.")

    # Actual baseline correction for Sound data buffer
    eeg_samples = eeg_samples - baseline_correction
    if not np.isfinite(eeg_samples).all() or not np.any(eeg_samples):
        raise ValueError("SOUND requires finite, nonzero baseline-corrected data.")

    # Smooth sigmas update coeff:
    sigmas_update_coeff = 0.05

    # Performs the SOUND algorithm for a given data.

    data = eeg_samples.T

    start = time.time()

    n0, _ = data.shape
    data = np.reshape(data, (n0, -1))

    LL = lfm @ lfm.T
    dn = np.empty((iterations, 1)) # Empty vector for convergences

    #################### Run beamformer SOUND #####################################################
    # See Metsomaa et al. 2024 Brain Topography for equations

    dataCov = np.matmul(data, data.T) / data.shape[1] # Estimate the data covariance matrix (sample covariance in the paper)

    # Estimate the neuronal covariance
    LL = lfm @ (lfm.T)
    regularization_term = lambda0*np.trace(LL) / num_of_channels
    LL_reg = LL / regularization_term

    # Save the previous sigma values before the new iteration:
    sigmas_prev_update = sigmas[:, np.newaxis].copy()

    # Iterate over channels
    n_iters = iterations
    convergence_reached = False
    for k in range(iterations):
        # Save the previous sigma values
        sigmas_old = np.copy(sigmas)
        # Update noise estimate values
        GAMMA = np.linalg.pinv(LL_reg + np.diagflat(np.square(sigmas))) # Eq. 18 in Metsomaa et al. 2024
        sigmas = np.array([(GAMMA[:, i] / GAMMA[i, i]).T @ (dataCov @ (GAMMA[:, i] / GAMMA[i, i])) for i in range(num_of_channels)]) # Eq. 20 in Metsomaa et al. 2024
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError("SOUND could not estimate positive finite channel noise.")

        # Following and storing the convergence of the algorithm
        max_noise_estimate_change = np.max(np.abs(sigmas_old - sigmas) / sigmas_old)
        #print("Output: Max noise estimate change = {}".format(max_noise_estimate_change))
        if max_noise_estimate_change < convergence_boundary and not fixed_iters: # terminates the iteration if the convergence boundary is reached
            print("Output: Convergence reached after {} iterations!".format(k+1))
            n_iters = k+1
            convergence_reached = True
            break

    # Make sure sigmas is a numpy array:
    sigmas = np.array(sigmas, dtype=np.float32)
    sigmas = np.expand_dims(sigmas,axis = 1)
    # Change sigmas smoothly:
    sigmas = sigmas_update_coeff*sigmas + (1 - sigmas_update_coeff)*sigmas_prev_update

    # Final data correction based on the final noise-covariance estimate.
    # Calculates matrices needed for SOUND spatial filter (for other functions)
    #SOUND_filter = LL @ np.linalg.inv(LL + regularization_term*C_noise)
    W = np.diag(1.0 / np.squeeze(sigmas))
    WL = np.matmul(W, lfm)
    WLLW = np.matmul(WL, WL.T)
    C = (WLLW + lambda0 * np.trace(WLLW) / num_of_channels * np.eye(num_of_channels))
    SOUND_filter = np.matmul(lfm, np.matmul(WL.T, np.linalg.solve(C, W)))

    # find the best-quality channel
    best_ch = np.argmin(sigmas)
    # Calculate the relative error in the best channel caused by SOUND overcorrection:
    rel_err = np.linalg.norm(SOUND_filter[best_ch,:]@data - data[best_ch,:])/np.linalg.norm(data[best_ch,:])
    print("Output: Relative error in best channel = {}".format(rel_err))

    end = time.time()
    print("Output: SOUND update time = {:.1f} ms".format(10 ** 3 * (end-start)))

    return SOUND_filter, sigmas, n_iters, convergence_reached
