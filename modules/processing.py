import numpy as np
import mne

def cluster_epochs_by_roi(epochs, roi_dict):
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
        roi_data = np.mean(data[:, ch_indices, :], axis=1)
        
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