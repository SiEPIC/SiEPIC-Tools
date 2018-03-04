# Setup Lumerical-Python integration, and load the SiEPIC-Tools Lumerical functions
import lumerical
import lumerical.load_lumapi
lumapi = lumerical.load_lumapi.LUMAPI

# for debugging, to reload the lumerical module:\n",
if 0:
    import sys
    if int(sys.version[0]) > 2:
      from importlib import reload
    reload(lumerical.interconnect)
    reload(lumerical.load_lumapi)

# Start Lumerical INTERCONNECT\n",
lumerical.interconnect.run_INTC()
INTC = lumerical.interconnect.INTC
lumapi.evalScript(INTC, "?'Test';")

# Find path to the MZI example netlist files
import os, inspect
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
folder = os.path.join(path,"../MZI")

# Perform Lumerical INTERCONNECT simulation
if 0:
  lumerical.interconnect.circuit_simulation(circuit_name="MZI", folder=folder, num_detectors=1, matlab_data_files=[], simulate=True, verbose=False)

if 1:
  lumerical.interconnect.circuit_simulation_monte_carlo(circuit_name="MZI", folder=folder, num_detectors=1, matlab_data_files=[], simulate=True, verbose=False)

