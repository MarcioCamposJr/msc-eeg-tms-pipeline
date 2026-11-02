# SOUND and SSP-SIR from PRIME

Source: https://github.com/mackelab/PRIME, commit
`254da3bbc4eae232c61a49789c84d985194efc2d` (local checkout).

- `sound.py`: `preprocessing/prep_data/sound_modified.py`.
- `ssp_sir.py`: `preprocessing/prep_data/ssp_sir_python.py`.

Original authors and references remain in the files. No explicit license was
found in the source checkout; this attribution does not grant a license or
establish permission to redistribute the original code.

The source SOUND implementation is marked experimental and not fully validated.
It uses the beamformer formulation referenced in Metsomaa et al. (2024), and is
not numerically interchangeable with the previous TESA SOUND implementation.
The original notice asks experimental users to contact its authors.

Local changes validate array shapes, finite values, parameter ranges and usable
matrix rank, handle absent artifact energy, and fix the noise-estimate shape in
the SOUND convergence check. The underlying update and reconstruction equations
are retained. SSP-SIR supports the source automatic window in milliseconds and
the source component-selection methods; no manual time-window API is implied.

The TEP template estimates the filters from the post-stimulus average and applies
them to individual EEG epochs without a MATLAB conversion. The SSP-SIR artifact
kernel is zero before stimulation. Both spatial reconstructions also apply to
the baseline. EMG and EOG are not passed to these routines.

The template computes an fsaverage forward model for the current EEG channels,
using the template's standard_1020 montage and MNE's built-in fsaverage transform.
As in PRIME, it uses the ico-5 cortical source space and the three-layer
fsaverage BEM solution. The original PRIME matrix must not be reused with a
different channel montage or channel order.

`mne.datasets.fetch_fsaverage` downloads missing anatomy/model files on the first
run into `data/models/fsaverage` and reuses them on subsequent runs. The download
cell runs at the start of the preprocessing notebook, before reading the BDF or
fitting ICA. Run the notebook from the beginning; no manual download is required.
Internet is needed when model files are missing. This directory
is already excluded from Git by the project's `data/` rule. No spherical fallback
is used if a download or forward computation fails. The cleaning metadata records
`forward_model="fsaverage"`.

This is template anatomy with standard electrode positions, not an individual
head model. Electrode placement relative to the template must be checked when
changing to measured sensor positions or another montage. See the MNE example:
https://mne.tools/1.12/auto_tutorials/forward/35_eeg_no_mri.html

The template retains the full epoch, including N100 and P200, rather than adopting
PRIME's 100 ms post-stimulus crop.
