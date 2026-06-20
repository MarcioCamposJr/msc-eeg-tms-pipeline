trigger_code= {
    "rest": 1,
    "prep": 2,
    "task_right": 3,
    "task_left": 4,
    "task_bilateral": 5,
    "tms_pulse_late": 10,
}

states_protocol = ["rest", "prep", "task_right", "task_left", "task_bilateral"]

tep_components_windows = {
    "N15": [0.010, 0.020],
    "P30": [0.025, 0.035],
    "N45": [0.035, 0.055],
    "P60": [0.055, 0.075],
    "N100": [0.090, 0.125],
    "P200": [0.150, 0.220]
}


# Dicionário de 20 ROIs otimizado para tarefas motoras e TMS
eeg_motor_rois = {
    # 1. Planejamento e Controle Executivo (Prefrontal)
    'Prefrontal_L': ['Fp1', 'AF3', 'AF7', 'F7'],
    'Prefrontal_R': ['Fp2', 'AF4', 'AF8', 'F8'],
    'Prefrontal_M': ['Fpz', 'AFz', 'Fz'],
    
    # 2. Área Motora Suplementar (SMA)
    # Fundamental para planejamento de movimentos autogerados e tarefas bilaterais
    'SMA_Midline': ['FCz', 'F1', 'F2'],
    
    # 3. Córtex Pré-Motor (PMC)
    # Preparação motora e movimentos guiados por estímulos externos
    'Premotor_L': ['FC3', 'FC5', 'F3', 'F5'],
    'Premotor_R': ['FC4', 'FC6', 'F4', 'F6'],
    
    # 4. Córtex Motor Primário (M1)
    # O alvo direto da TMS e responsável pela execução motora (mãos/braços)
    'M1_L': ['C3', 'C1', 'C5'],
    'M1_R': ['C4', 'C2', 'C6'],
    'M1_M': ['Cz'], # Foco em membros inferiores/tronco
    
    # 5. Córtex Somatossensorial Primário (S1)
    # Processa o feedback proprioceptivo após o movimento ou o pulso TMS
    'S1_L': ['CP3', 'CP1', 'CP5'],
    'S1_R': ['CP4', 'CP2', 'CP6'],
    'S1_M': ['CPz'],
    
    # 6. Córtex Parietal Posterior (Integração Sensório-Motora)
    # Importante para a transformação visuo-motora
    'Parietal_L': ['P1', 'P3', 'P5', 'P7'],
    'Parietal_R': ['P2', 'P4', 'P6', 'P8'],
    'Parietal_M': ['Pz'],
    
    # 7. Córtex Occipital (Visual) - Agrupamento amplo
    'Occipital_L': ['O1', 'PO3', 'PO7', 'PO9'],
    'Occipital_R': ['O2', 'PO4', 'PO8', 'PO10'],
    'Occipital_M': ['Oz', 'POz', 'Iz'],
    
    # 8. Córtex Temporal (Auditivo/Linguagem) - Agrupamento amplo
    'Temporal_L': ['FT7', 'FT9', 'T7', 'T9', 'TP7', 'TP9'],
    'Temporal_R': ['FT8', 'FT10', 'T8', 'T10', 'TP8', 'TP10']
}