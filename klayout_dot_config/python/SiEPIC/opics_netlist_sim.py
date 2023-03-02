import pya

def circuit_simulation_opics(verbose=False,opt_in_selection_text=[], require_save=True):
    ''' Simulate the circuit using OPICS
    Using a netlist extracte from the layout'''
    
    # obtain the spice file from the layout
    from SiEPIC.netlist import export_spice_layoutview
    spice_filepath, _ = export_spice_layoutview(verbose=False,opt_in_selection_text=[], require_save=require_save)
    
    from SiEPIC.opics import libraries
    from SiEPIC.opics.network import Network
    from SiEPIC.opics.utils import netlistParser, NetlistProcessor
    from SiEPIC.opics.globals import C as c_
        
    print(spice_filepath)
    
    # get netlist data
    circuitData = netlistParser(spice_filepath).readfile()
    
    print(circuitData)
    
    # process netlist data
    subckt = NetlistProcessor(spice_filepath, Network, libraries, c_, circuitData, verbose=False)
    
    print(subckt)
    
    # simulate network
    subckt.simulate_network()
    
    # get input and output net labels
    inp_idx = subckt.global_netlist[list(subckt.global_netlist.keys())[-1]].index(
        circuitData["inp_net"]
    )
    out_idx = [
        subckt.global_netlist[list(subckt.global_netlist.keys())[-1]].index(each)
        for each in circuitData["out_net"]
    ]
    
    ports = [[each_output, inp_idx] for each_output in out_idx]
    
    
    # plot results
    # subckt.sim_result.plot_sparameters(ports=ports, interactive=False)
    
    
    # Plot using Plotly:
    import plotly.express as px
    import pandas as pd # https://pandas.pydata.org/docs/user_guide/10min.html
    result = subckt.sim_result.get_data()
    wavelengths = c_/subckt.sim_result.f
    transmission = result['S_0_1']
    reflection = result['S_0_0']

    # *** There is something wrong where the f vector has a different length
    # than the results vector. This fixes it:
    import numpy as np
    freq = np.linspace(subckt.sim_result.f[0], subckt.sim_result.f[-1], len(transmission))
    wavelengths = c_/freq

    
    # Single line:
    #df = pd.DataFrame(transmission, index=wavelengths, columns=['Transmission'])
    
    # Two lines:
    import numpy as np
    df = pd.DataFrame(np.stack((transmission, reflection)).transpose(), index=wavelengths, columns=['Transmission','Reflections'])
    fig = px.line(df, labels={'index':'Wavelength', 'value':'Transmission (dB)'}, markers=True)
    fig.show()
    
    
