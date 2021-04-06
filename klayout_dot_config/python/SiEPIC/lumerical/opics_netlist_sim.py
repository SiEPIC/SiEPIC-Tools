import time
import os, sys
import pathlib
from opics import libraries
from opics.network import Network
from opics.utils import netlistParser, NetlistProcessor
from opics.globals import c as c_

# warnings.filterwarnings('ignore') #ignore all/complex number warnings from numpy or scipy

sim_start = time.time()

print(sys.argv[1])
# read netlist
spice_filepath = sys.argv[1] #pathlib.Path(os.path.dirname(__file__) + r"\\test_mzi.spi")

print(spice_filepath)

# get netlist data
circuitData = netlistParser(spice_filepath).readfile()

#print(circuitData)

# process netlist data
subckt = NetlistProcessor(spice_filepath, Network, libraries, c_, circuitData, verbose=False)

# simulate network
subckt.simulate_network()

# get input and output net labels
inp_idx = subckt.sim_result.nets[0].index(circuitData["inp_net"])
out_idx = [subckt.sim_result.nets[0].index(each) for each in circuitData["out_net"]]

ports = [[each_output, inp_idx] for each_output in out_idx]

# plot results
subckt.sim_result.plot_sparameters(ports=ports)