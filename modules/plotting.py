import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
import mne

from config import tep_components_windows

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

def plot_tep_components_topomap(epochs, method='evoked', vlim=(None, None), sphere=None):
    """
    Plota um topomap para cada componente do TEP (Potencial Evocado por TMS)
    baseado na Área sob a Curva (AUC) do sinal retificado em janelas temporais específicas.

    Parâmetros:
    - epochs: objeto mne.Epochs.
    - components_config: dict. Chaves são os nomes das componentes (ex: 'N15') 
                         e os valores são listas/tuplas com [tmin, tmax] em segundos.
    - method: 'evoked' (calcula AUC do sinal médio) ou 'epochs' (calcula AUC por época e média).
    - vlim: tupla (min, max). Controla os limites da escala de cores (colorbar).
            Ex: (0, 1e-5) ou (None, None) para cálculo automático.
    - sphere: controla o ajuste da "cabeça fictícia" aos sensores. 
              Pode ser um float (raio), 'eeglab' ou uma tupla com centro e raio.
    """


    n_components = len(tep_components_windows)
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    
    if n_components == 1:
        axes = [axes]
        
    sfreq = epochs.info['sfreq']
    
    if method == 'evoked':
        # Calcula o evoked (média através das épocas) primeiro
        evoked = epochs.average()
        
    for ax, (comp_name, time_window) in zip(axes, tep_components_windows.items()):
        tmin, tmax = time_window
        
        if method == 'evoked':
            # Recorta o sinal evoked para a janela de tempo da componente
            evoked_cropped = evoked.copy().crop(tmin=tmin, tmax=tmax)
            # Retifica o sinal (valor absoluto)
            rectified_data = np.abs(evoked_cropped.data)
            # Calcula a AUC usando a regra do trapézio
            auc = trapezoid(rectified_data, dx=1/sfreq, axis=-1)
            info = evoked_cropped.info
        elif method == 'epochs':
            # Recorta as épocas para a janela de tempo
            epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
            # Retifica o sinal de todas as épocas
            rectified_data = np.abs(epochs_cropped.get_data())
            # Calcula a AUC para cada época e tira a média entre elas
            auc_per_trial = trapezoid(rectified_data, dx=1/sfreq, axis=-1)
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
            extrapolate='head',
            vlim=vlim,     # <-- Adicionado para controle da colorbar
            sphere=sphere  # <-- Adicionado para melhorar ajuste da cabeça
        )
        
        ax.set_title(f"{comp_name}\n({tmin*1000:.0f}-{tmax*1000:.0f} ms)")
        
        # Adiciona a barra de cores
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, label="AUC (V·s)")

    plt.tight_layout()
    plt.show()
    return fig

def plot_tep(epochs):
    evoked = epochs.average()
    media_canais = np.mean(evoked.data, axis=0)

    # Configurações globais de estilo para publicação (limpo e legível)
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.spines.top': False,   # Remove a borda superior
        'axes.spines.right': False, # Remove a borda direita
        'font.family': 'sans-serif' # Fontes sem serifa são melhores para slides
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    # 2. Plota o sinal TEP com linha mais espessa e cor sólida escura
    ax.plot(evoked.times, media_canais * 1e6, color="#2E2E2E", linewidth=2.5)

    # 3. Sombreamento e anotações dos componentes
    for componente, (inicio, fim) in tep_components_windows.items():
        # Sombreamento padronizado em um cinza-azulado elegante
        ax.axvspan(inicio, fim, color="#CA8585", alpha=0.4, zorder=0)
        
        # Adiciona o texto do componente na parte superior da banda (Y=0.95 ou 95% da altura)
        ax.text((inicio + fim) / 2, 0.95, componente, 
                transform=ax.get_xaxis_transform(), 
                horizontalalignment='center', 
                fontsize=20, fontweight='bold', color="#C73030")

    # 4. Eixos e marcação do pulso TMS
    ax.set_xlabel("Time (s)", fontweight='bold', labelpad=10)
    ax.set_ylabel("Amplitude (µV)", fontweight='bold', labelpad=10)

    # Linha do tempo zero destacada em vermelho escuro
    ax.axvline(0, color='#D32F2F', linestyle='--', linewidth=2, label='TMS Pulse')

    # Linha horizontal no zero (baseline)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)

    # Adiciona um grid sutil apenas no eixo Y para guiar o olhar na amplitude
    ax.yaxis.grid(True, alpha=0.3, linestyle=':')

    # Apenas a legenda do pulso TMS em um canto discreto
    ax.legend(loc='lower right', frameon=False, fontsize=20)

    plt.tight_layout()
    plt.show()

def plot_tep_conditions(evokeds_prep_dict, evokeds_task_dict, evokeds_rest_dict, times, roi_ch,  cores = None):
    if cores is None:
        cores = {
            'Right': 'tab:blue',
            'Left': 'tab:orange',
            'Bilateral': 'tab:green'
        }

    estados = [
        ('Resting', evokeds_rest_dict),
        ('Preparation', evokeds_prep_dict),
        ('Execution', evokeds_task_dict)
    ]
    
    ch_names = ''
    for ch in roi_ch:
        ch_names += ch + ', '
    ch_names = ch_names[:-2]

    # Criar a figura com 3 subplots verticais
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Iterar sobre cada estado (Repouso, Preparação, Tarefa) e seu respectivo eixo
    for ax, (nome_estado, dict_estado) in zip(axes, estados):
        
        # Iterar sobre as 3 condições dentro daquele estado
        for nome_condicao, dados in dict_estado.items():
            
            # O nome da condição base (Right, Left ou Bilateral) para escolher a cor
            condicao_base = nome_condicao.split()[0]
            cor = cores.get(condicao_base, 'gray')
            
            # Converter para microvolts (µV) para facilitar a visualização (opcional)
            # O MNE por padrão exporta EEG em Volts (V). Se já estiver em µV, remova o "* 1e6"
            dados_uv = dados * 1e6 
            
            # Calcular média e desvio padrão ao longo dos canais (axis=0)
            media_canais = np.mean(dados_uv, axis=0)
            dp_canais = np.std(dados_uv, axis=0)
            
            # Plotar a linha da média
            ax.plot(times, media_canais, label=nome_condicao, color=cor, linewidth=2)
            
            # Plotar a área sombreada representando o desvio padrão
            ax.fill_between(times, 
                            media_canais - dp_canais, 
                            media_canais + dp_canais, 
                            color=cor, alpha=0.2) # alpha controla a transparência
            
        # Formatação do subplot
        ax.set_title(f'TMS evoked potencial ({ch_names}) - {nome_estado}', fontsize=18)
        ax.set_ylabel('Amplitude (µV)', fontsize=16)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7) # Linha no t=0
        ax.legend(loc='upper right', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Adicionar o rótulo do eixo X apenas no último gráfico
    axes[-1].set_xlabel('Time (s)', fontsize=16)

    plt.tight_layout()
    plt.show()