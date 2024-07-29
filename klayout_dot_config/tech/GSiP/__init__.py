print('SiEPIC-GSiP PDK Python module: siepic_gsip_pdk, KLayout technology: GSipP')

# Load the KLayout technology, when running in Script mode
import pya
import os
tech = pya.Technology().create_technology('GSiP')
tech.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'GSiP.lyt'))

# then import all the technology modules
from . import pymacros


