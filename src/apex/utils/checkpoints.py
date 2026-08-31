import re
from collections import OrderedDict


def compatible_state_dict(state_dict):
    comp = OrderedDict()
    for key, value in state_dict.items():
        new_key = re.sub(r'conv(1|s\.[0-9]+)\.weight',
                         r'conv\1.lin.weight', key)
        comp[new_key] = value.T if new_key != key else value
    return comp
