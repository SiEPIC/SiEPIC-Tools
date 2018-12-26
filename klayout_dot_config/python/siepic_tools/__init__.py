# Detecting environment

try:
    import pya
    is_GSI = pya.__spec__.origin == 'built-in'
except ImportError:
    is_GSI = False
