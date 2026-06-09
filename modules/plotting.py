import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_boxplot_with_points(data, labels):
    # 2. Organizando os dados em um DataFrame
    # Criamos uma coluna com os valores e outra com o nome de cada classe

    if len(data) != len(labels):
        raise ValueError("O número de arrays de dados deve ser igual ao número de rótulos.")

    df = pd.DataFrame({
    'Valores': np.concatenate(data),
    'Classe': [label for i, label in enumerate(labels) for _ in range(len(data[i]))]
    })

    # 3. Configuração de Estilo
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 4. Boxplot Elegante
    # 'fliersize=0' esconde os outliers padrão para não dobrar com o stripplot
    ax = sns.boxplot(
        x='Classe', y='Valores', data=df,
        hue='Classe',        
        legend=False,         
        linewidth=1.5,
        width=0.5,
        fliersize=0
    )

    # 5. Camada de Pontos (Stripplot)
    # O 'jitter=True' espalha os pontos lateralmente para não ficarem sobrepostos
    sns.stripplot(
        x='Classe', y='Valores', data=df,
        color=".3",           # Cinza escuro para os pontos
        size=4,               # Pontos pequenos e delicados
        alpha=0.3,            # Transparência para dar profundidade
        jitter=True
    )

    # 6. Detalhes Finais (Títulos e Eixos)
    plt.title("Comparação de Distribuição entre Classes", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Categorias", fontsize=12)
    plt.ylabel("Mensuração (Unidade)", fontsize=12)

    # Remove as bordas superiores e laterais (despine)
    sns.despine(offset=10, trim=True)

    plt.tight_layout()
    plt.show()

import mne

def plot_tep_components_topomap(epochs, components_config, method='evoked'):
    """
    Plota um topomap para cada componente do TEP (Potencial Evocado por TMS)
    baseado na Área sob a Curva (AUC) do sinal retificado em janelas temporais específicas.

    Parâmetros:
    - epochs: objeto mne.Epochs.
    - components_config: dict. Chaves são os nomes das componentes (ex: 'N15') 
                         e os valores são listas/tuplas com [tmin, tmax] em segundos.
                         Exemplo: {"N15": [0.010, 0.020], "P30": [0.025, 0.035]}
    - method: 'evoked' (calcula AUC do sinal médio) ou 'epochs' (calcula AUC por época e faz a média).
    """
    n_components = len(components_config)
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    
    if n_components == 1:
        axes = [axes]
        
    sfreq = epochs.info['sfreq']
    
    if method == 'evoked':
        # Calcula o evoked (média através das épocas) primeiro
        evoked = epochs.average()
        
    for ax, (comp_name, time_window) in zip(axes, components_config.items()):
        tmin, tmax = time_window
        
        if method == 'evoked':
            # Recorta o sinal evoked para a janela de tempo da componente
            evoked_cropped = evoked.copy().crop(tmin=tmin, tmax=tmax)
            # Retifica o sinal (valor absoluto)
            rectified_data = np.abs(evoked_cropped.data)
            # Calcula a AUC usando a regra do trapézio
            auc = np.trapz(rectified_data, dx=1/sfreq, axis=-1)
            info = evoked_cropped.info
        elif method == 'epochs':
            # Recorta as épocas para a janela de tempo
            epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
            # Retifica o sinal de todas as épocas
            rectified_data = np.abs(epochs_cropped.get_data())
            # Calcula a AUC para cada época e tira a média entre elas
            auc_per_trial = np.trapz(rectified_data, dx=1/sfreq, axis=-1)
            auc = np.mean(auc_per_trial, axis=0)
            info = epochs_cropped.info
        else:
            raise ValueError("O método deve ser 'evoked' ou 'epochs'.")
            
        # Plota o topomap
        im, _ = mne.viz.plot_topomap(
            auc, 
            info, 
            axes=ax, 
            show=False,
            cmap='Reds', # Colormap sequencial pois AUC é sempre positiva
            extrapolate='local'
        )
        
        ax.set_title(f"{comp_name}\n({tmin*1000:.0f}-{tmax*1000:.0f} ms)")
        
        # Adiciona a barra de cores
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, label="AUC (V·s)")

    plt.tight_layout()
    plt.show()
    return fig