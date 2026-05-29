from config import trigger_code

def decode_8bit_trigger(trigger_value):

    template_8bit = "8Bit "

    for key, value in trigger_code.items():
        value_8bit = template_8bit + str(value)
        if value_8bit == trigger_value:
            return key
        elif trigger_value == "Stimulus A":
            return "tms_pulse"
    return "Unknown Trigger"

def convert_dict_trigger(dict, function= None):
    function = decode_8bit_trigger if function is None else function
    converted_dict = {}
    for key, value in dict.items():
        converted_key = function(key)
        converted_dict[converted_key] = value
    return converted_dict