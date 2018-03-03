path = "/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Python"

import os
os.environ['PATH'] += ':/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/MacOS' 
os.environ['PATH'] += ':/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Python'
os.environ['PATH'] += ':/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Matlab'
import lumapi
        