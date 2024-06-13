# Some helpers for serializing data. This is not dealing with general data but only with the necessary types encountered
# in this module.
import numpy as np


def serialize_dict_values(d: dict) -> dict:
    """ Convert all values in a dict to a serializeable type """

    # numpy data to lists for serialization
    ser_results = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            ser_results[k] = serialize_list(v.tolist())
        elif isinstance(v, np.int32):
            ser_results[k] = int(v)
        elif isinstance(v, np.float64):
            ser_results[k] = float(v)
        elif isinstance(v, list):
            ser_results[k] = serialize_list(v)
        else:
            ser_results[k] = v

    return ser_results

def serialize_list(li: list) -> list:
    ser_results = []
    for e in li:
        if isinstance(e, np.ndarray):
            ser_results.append(e.tolist())
        elif isinstance(e, np.int32):
            ser_results.append(int(e))
        elif isinstance(e, np.float64):
            ser_results.append(float(e))
        else:
            ser_results.append(e)

    return ser_results
