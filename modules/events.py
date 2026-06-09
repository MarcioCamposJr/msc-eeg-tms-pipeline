import numpy as np

from config import states_protocol

def get_events_time(events, event_id, s_freq):
    rest = []
    prep = []
    task = []

    state_event_ids = [event_id[state] for state in states_protocol]
    task_event_ids = [
        event_id["task_right"],
        event_id["task_left"],
        event_id["task_bilateral"],
    ]
    state_name_by_id = {event_id[state]: state for state in states_protocol}

    for id, event in enumerate(events):
        if event[2] not in state_event_ids:
            continue

        next_state = None
        next_state_id = None
        for next_id in range(id + 1, len(events)):
            if events[next_id][2] in state_event_ids:
                next_state = events[next_id]
                next_state_id = next_id
                break

        if next_state is None:
            continue

        time = (next_state[0] - event[0]) * (1 / s_freq)
        if time > 20:
            print(
                f"Evento {event[2]} com tempo de {time} segundos, "
                f"id: {id}, proximo estado id: {next_state_id}"
            )
            continue

        event_name = state_name_by_id[event[2]]
        next_state_name = state_name_by_id[next_state[2]]

        if event_name == "rest" and next_state_name == "prep":
            rest.append(time)
        elif event_name == "prep" and next_state[2] in task_event_ids:
            prep.append(time)
        elif event[2] in task_event_ids and next_state_name == "rest":
            task.append(time)

    rest = np.array(rest)
    prep = np.array(prep)
    task = np.array(task)

    return rest, prep, task

def get_events_time_tms(events, event_id, s_freq):
    tms_pulse_interval_rest_prep = []
    tms_pulse_interval_prep_task = []
    tms_pulse_interval_task_rest = []

    state_event_ids = [event_id[state] for state in states_protocol]
    task_event_ids = [
        event_id["task_right"],
        event_id["task_left"],
        event_id["task_bilateral"],
    ]
    state_name_by_id = {event_id[state]: state for state in states_protocol}

    def get_previous_state(event_index):
        for state_id in range(event_index - 1, -1, -1):
            if events[state_id][2] in state_event_ids:
                return events[state_id], state_id
        return None, None

    for id, event in enumerate(events):
        if event[2] != event_id["tms_pulse"]:
            continue

        previous_state, previous_state_id = get_previous_state(id)

        next_pulse = None
        next_pulse_id = None
        for next_id in range(id + 1, len(events)):
            if events[next_id][2] == event_id["tms_pulse"]:
                next_pulse = events[next_id]
                next_pulse_id = next_id
                break

        if previous_state is None or next_pulse is None:
            continue

        next_pulse_state, next_pulse_state_id = get_previous_state(next_pulse_id)
        if next_pulse_state is None:
            continue

        time = (next_pulse[0] - event[0]) * (1 / s_freq)
        if time > 20:
            print(
                f"Pulso TMS com intervalo de {time} segundos, "
                f"id: {id}, estado anterior id: {previous_state_id}, "
                f"proximo pulso id: {next_pulse_id}, "
                f"estado do proximo pulso id: {next_pulse_state_id}"
            )
            continue

        previous_state_name = state_name_by_id[previous_state[2]]
        next_pulse_state_name = state_name_by_id[next_pulse_state[2]]

        if previous_state_name == "rest" and next_pulse_state_name == "prep":
            tms_pulse_interval_rest_prep.append(time)
        elif previous_state_name == "prep" and next_pulse_state[2] in task_event_ids:
            tms_pulse_interval_prep_task.append(time)
        elif previous_state[2] in task_event_ids and next_pulse_state_name == "rest":
            tms_pulse_interval_task_rest.append(time)

    tms_pulse_interval_rest_prep = np.array(tms_pulse_interval_rest_prep)
    tms_pulse_interval_prep_task = np.array(tms_pulse_interval_prep_task)
    tms_pulse_interval_task_rest = np.array(tms_pulse_interval_task_rest)

    return tms_pulse_interval_rest_prep, tms_pulse_interval_prep_task, tms_pulse_interval_task_rest


def get_pulse_time_in_state(events, event_id, s_freq):
    pulse_time_rest = []
    pulse_time_prep = []
    pulse_time_task = []

    state_event_ids = [event_id[state] for state in states_protocol]
    task_event_ids = [
        event_id["task_right"],
        event_id["task_left"],
        event_id["task_bilateral"],
    ]

    def get_previous_state(event_index):
        for state_id in range(event_index - 1, -1, -1):
            if events[state_id][2] in state_event_ids:
                return events[state_id], state_id
        return None, None

    for id, event in enumerate(events):
        if event[2] != event_id["tms_pulse"]:
            continue

        previous_state, previous_state_id = get_previous_state(id)
        if previous_state is None:
            continue
            
        time_in_state = (event[0] - previous_state[0]) * (1 / s_freq)
        
        state_id_val = previous_state[2]
        
        if state_id_val == event_id["rest"]:
            pulse_time_rest.append(time_in_state)
        elif state_id_val == event_id["prep"]:
            pulse_time_prep.append(time_in_state)
        elif state_id_val in task_event_ids:
            pulse_time_task.append(time_in_state)
            
    return np.array(pulse_time_rest), np.array(pulse_time_prep), np.array(pulse_time_task)


def get_events_tms_per_task(events, event_id):
    tms_pulses = []
    events_id = {
        "tms_pulse_task_right": 1,
        "tms_pulse_task_left": 2,
        "tms_pulse_task_bilateral": 3,
        "tms_pulse_prep_right": 4,
        "tms_pulse_prep_left": 5,
        "tms_pulse_prep_bilateral": 6,
        "tms_pulse_rest_right": 7,
        "tms_pulse_rest_left": 8,
        "tms_pulse_rest_bilateral": 9,
    }

    state_event_ids = [event_id[state] for state in states_protocol]
    task_event_ids = [
        event_id["task_right"],
        event_id["task_left"],
        event_id["task_bilateral"],
    ]

    def get_previous_state(event_index):
        for state_id in range(event_index - 1, -1, -1):
            if events[state_id][2] in state_event_ids:
                return events[state_id], state_id
        return None, None

    def get_associated_task(event_index):
        for state_id in range(event_index, len(events)):
            if events[state_id][2] in task_event_ids:
                return events[state_id][2]
        return None

    for id, event in enumerate(events):
        if event[2] != event_id["tms_pulse"]:
            continue

        previous_state, previous_state_id = get_previous_state(id)
        if previous_state is None:
            continue

        state_id_val = previous_state[2]

        if state_id_val in task_event_ids:
            associated_task_id = state_id_val
        else:
            associated_task_id = get_associated_task(id)

        if associated_task_id is None:
            continue

        task_suffix = ""
        if associated_task_id == event_id["task_right"]:
            task_suffix = "right"
        elif associated_task_id == event_id["task_left"]:
            task_suffix = "left"
        elif associated_task_id == event_id["task_bilateral"]:
            task_suffix = "bilateral"

        state_id_val = previous_state[2]

        if state_id_val in task_event_ids:
            tms_pulses.append([event[0], 0, events_id[f"tms_pulse_task_{task_suffix}"]])
        elif state_id_val == event_id["prep"]:
            tms_pulses.append([event[0], 0, events_id[f"tms_pulse_prep_{task_suffix}"]])
        elif state_id_val == event_id["rest"]:
            tms_pulses.append([event[0], 0, events_id[f"tms_pulse_rest_{task_suffix}"]])

    return np.array(tms_pulses), events_id