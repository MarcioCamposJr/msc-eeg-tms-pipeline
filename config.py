trigger_code= {
    "rest": 1,
    "prep": 2,
    "task_right": 3,
    "task_left": 4,
    "task_bilateral": 5,
    "tms_pulse_late": 10,
}

states_protocol = ["rest", "prep", "task_right", "task_left", "task_bilateral"]

# Configuração das janelas temporais das componentes do TEP (em segundos)
tep_components_windows = {
    "N15": [0.010, 0.020],
    "P30": [0.025, 0.035],
    "N45": [0.040, 0.050],
    "P60": [0.055, 0.065],
    "N100": [0.090, 0.110],
    "P200": [0.180, 0.220]
}