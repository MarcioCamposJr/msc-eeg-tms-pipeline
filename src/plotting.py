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