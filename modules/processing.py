import numpy as np
import mne
from scipy.integrate import trapezoid
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter

from utils.validation_data import check_hand
from config import tep_components_windows

def cluster_epochs_by_roi(epochs, roi_dict, method='mean'):
    # Extrair os dados: formato (n_epocas, n_canais, n_tempos)
    data = epochs.get_data(copy=True)
    ch_names = epochs.ch_names
    
    new_data = []
    new_ch_names = []

    for roi_name, channels in roi_dict.items():

        # Encontrar os índices dos canais do dicionário que realmente existem nos seus dados
        ch_indices = [ch_names.index(ch) for ch in channels if ch in ch_names]
        
        if not ch_indices:
            print(f"Aviso: Nenhum canal do grupo {roi_name} foi encontrado nos dados.")
            continue
            
        # Tirar a média dos canais deste cluster ao longo do eixo 1 (Canais)
        # O resultado será (n_epocas, n_tempos)
        if method == 'mean':
            roi_data = np.mean(data[:, ch_indices, :], axis=1)
        elif method == 'PCA':
            if len(ch_indices) > 1:
                pca = PCA(n_components=1)                
                spatial_filter = UnsupervisedSpatialFilter(pca, average=False)
                pca_data_3d = spatial_filter.fit_transform(data[:, ch_indices, :])
                roi_data = pca_data_3d.squeeze(axis=1)
            else:
                roi_data = data[:, ch_indices, :].squeeze()
        
        new_data.append(roi_data)
        new_ch_names.append(roi_name)
        
    # Empilhar tudo de volta. Novo formato: (n_epocas, n_rois, n_tempos)
    new_data = np.stack(new_data, axis=1)
    
    # Criar um novo MNE Info para abrigar esses clusters
    new_info = mne.create_info(
        ch_names=new_ch_names, 
        sfreq=epochs.info['sfreq'], 
        ch_types='eeg'
    )
    
    # Adicionar metadados e criar o novo objeto Epochs
    new_epochs = mne.EpochsArray(
        new_data, 
        new_info, 
        tmin=epochs.tmin,
        events=epochs.events,
        event_id=epochs.event_id
    )
    
    return new_epochs

def get_data_evokeds_condition(epochs, roi, time_cropped):
    picks = [ch for ch in roi if ch in epochs.ch_names]
    tmin, tmax = time_cropped
    times = epochs.times[(epochs.times >= tmin) & (epochs.times <= tmax)]

    evokeds_prep_dict= {}
    evokeds_task_dict = {}
    evokeds_rest_dict = {}

    for state in list(epochs.event_id.keys()):
        if 'prep' in state:
            evokeds_prep_dict[check_hand(state)] = epochs[state].average().pick(picks).crop(time_cropped[0], time_cropped[1]).data
        elif 'task' in state:
            evokeds_task_dict[check_hand(state)] = epochs[state].average().pick(picks).crop(time_cropped[0], time_cropped[1]).data
        elif 'rest' in state:
            evokeds_rest_dict[check_hand(state)] = epochs[state].average().pick(picks).crop(time_cropped[0], time_cropped[1]).data

    return evokeds_prep_dict, evokeds_task_dict, evokeds_rest_dict, times
    
def get_AUC_tep_components(evokeds_dict, times, sfreq):

    def get_idx_time_window(times, tmin, tmax):
        idx_min = np.abs(times - tmin).argmin()
        idx_max = np.abs(times - tmax).argmin()
        return idx_min, idx_max
    
    auc_components = {}
    for state, data in evokeds_dict.items():
        auc_components[state] = {}
        for comp_name, time_window in tep_components_windows.items():
            tmin, tmax = time_window
            idx_min, idx_max = get_idx_time_window(times, tmin, tmax)
            media_canais = np.mean(data, axis=0)
            data_cropped = media_canais[idx_min:idx_max]
            dados_uv = data_cropped * 1e6 
            rectified_data = np.abs(dados_uv)
            auc = trapezoid(rectified_data, dx=1000/sfreq, axis=-1)
            auc_components[state][comp_name] = auc

    return auc_components