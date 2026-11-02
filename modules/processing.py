import numpy as np
import mne
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
from tepfeatures import TEPExtractor
import pandas as pd

from typing import Literal, Union, List, Dict

from utils.validation_data import check_hand
from config import eeg_motor_rois, events_tms_per_condition

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

class TEPAnalysis:
    def __init__(self, epochs, cluster: bool = True):
        self.epochs = epochs.copy()
        self.sfreq = epochs.info['sfreq']
        self.times = epochs.times

        self.drop_channels_not_used()

        if cluster:
            self.epochs = cluster_epochs_by_roi(self.epochs, roi_dict=eeg_motor_rois, method='PCA')
            print("Dimensões originais:", epochs.copy().get_data().shape)
            print("Novas dimensões (ROIs):", self.epochs.get_data().shape)

        self.epochs_per_condition = self.__split_conditions()
        self.tep_extractor_per_condition = self.__create_tep_extractor()

    def drop_channels_not_used(self, chanells_to_drop = ["EOG", "EMG L", "EMG R"]):
        try:
            self.epochs.drop_channels(chanells_to_drop)
        except ValueError:
            print("Aviso: Alguns canais de referência não foram encontrados e não puderam ser removidos.")

    def __split_conditions(self):
        conditions = {}
        for event_name in self.epochs.event_id:
            condition_epochs = self.epochs[event_name]
            conditions[event_name] = condition_epochs
        return conditions

    def __create_tep_extractor(self):
        tep_extractor_per_condition = {}
        for condition, epochs_condition in self.epochs_per_condition.items():
            print(f"Creating TEPExtractor for condition: {condition}")
            tep_extractor_per_condition[condition] = TEPExtractor(epochs_condition.average())
        return tep_extractor_per_condition

    def get_auc_tep_per_condition(self, condition_target: Literal['rest', 'prep', 'task']):
        df = None
        for condition in events_tms_per_condition.keys():
            if condition_target in condition:
                if df is None:
                    df = self.tep_extractor_per_condition[condition].temporal.get_area()
                    df['condition'] = condition
                else:
                    df_ = self.tep_extractor_per_condition[condition].temporal.get_area()
                    df_['condition'] = condition
                    df = pd.concat([df, df_])
        return df

    def get_band_power_per_condition(self, condition_target: Literal['rest', 'prep', 'task'], bands: Dict[str, List[float]]):
        df = None
        for condition in events_tms_per_condition.keys():
            if condition_target in condition:
                if df is None:
                    df = self.tep_extractor_per_condition[condition].spectral.get_band_power(bands)
                    df['condition'] = condition
                else:
                    df_ = self.tep_extractor_per_condition[condition].spectral.get_band_power(bands)
                    df_['condition'] = condition
                    df = pd.concat([df, df_])
        return df

    def get_significant_channels_snr(self, baseline: tuple[float, float], threshold: float = 3.0, conditions: Union[str, list[str], None] = None, peak_name: Union[str, list[str], None] = None):
        """
        Verifica quais canais possuem SNR significante (acima do limiar).

        Parameters:
        -----------
        baseline : tuple of float
            A janela de tempo para o baseline em segundos (ex: (-0.5, -0.01)).
        threshold : float
            Valor de limiar mínimo para a razão sinal-ruído (default: 3.0).
        conditions : str, list of str, ou None
            Condição(ões) a ser(em) avaliada(s). Se None, usa todas as disponíveis.
        peak_name : str, list of str, ou None
            Nome(s) do pico a ser(em) avaliado(s). Se None, avalia todos os definidos.

        Returns:
        --------
        pd.DataFrame
            Um DataFrame contendo os canais, uma coluna booleana 'is_significant'
            e uma coluna 'condition' indicando a qual condição se refere.
        """
        if conditions is None:
            target_conditions = list(self.tep_extractor_per_condition.keys())
        elif isinstance(conditions, str):
            target_conditions = [conditions]
        else:
            target_conditions = conditions

        results_list = []

        for condition in target_conditions:
            if condition not in self.tep_extractor_per_condition:
                print(f"Aviso: Condição '{condition}' não encontrada.")
                continue

            # Obtem os valores brutos de SNR (retorna df com colunas: channel, peak_name, snr...)
            df_snr = self.tep_extractor_per_condition[condition].temporal.get_snr(
                peak_name=peak_name,
                baseline=baseline
            )

            # Agrupa os dados pelo canal e pega o SNR máximo dele
            # naquele(s) pico(s).
            channel_max_snr = df_snr.groupby('channel')['snr'].max()

            # Converte para DataFrame
            df_significant = channel_max_snr.reset_index()
            # Cria a coluna booleana baseada no limiar
            df_significant['is_significant'] = df_significant['snr'] >= threshold
            df_significant['condition'] = condition

            results_list.append(df_significant)

        if results_list:
            return pd.concat(results_list, ignore_index=True)
        else:
            return pd.DataFrame()
