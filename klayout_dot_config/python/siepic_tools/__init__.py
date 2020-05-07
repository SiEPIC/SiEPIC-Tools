# Detecting environment

try:
    import pya
    is_GSI = pya.__package__ == ''
except ImportError:
    is_GSI = False
