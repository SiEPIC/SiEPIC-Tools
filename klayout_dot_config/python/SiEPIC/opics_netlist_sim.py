import pya

def circuit_simulation_opics(verbose=False,opt_in_selection_text=[], require_save=True):
    ''' Simulate the circuit using OPICS
    Using a netlist extracte from the layout'''
    
    # Required packages
    from SiEPIC.install import install
    if not install('plotly'):
        pya.MessageBox.warning(
        "Missing package", "The OPICS circuit simulator does not function without the package 'plotly'.",  pya.MessageBox.Ok)    
        return None
    if not install('pandas'):
        pya.MessageBox.warning(
        "Missing package", "The OPICS circuit simulator does not function without the package 'pandas'.",  pya.MessageBox.Ok)    
        return None
    if not install('packaging'):
        pya.MessageBox.warning(
        "Missing package", "The OPICS circuit simulator does not function without the package 'packaging'.",  pya.MessageBox.Ok)    
        return None
    if not install('defusedxml'):
        pya.MessageBox.warning(
        "Missing package", "The OPICS circuit simulator does not function without the package 'defusedxml'.",  pya.MessageBox.Ok)    
        return None
    if not install('numpy'):
        pya.MessageBox.warning(
        "Missing package", "The OPICS circuit simulator does not function without the package 'numpy'.",  pya.MessageBox.Ok)    
        return None
    if not install('yaml'):
        pya.MessageBox.warning(
        "Missing package", "The OPICS circuit simulator does not function without the package 'yaml'.",  pya.MessageBox.Ok)    
        return None
    if not install('scipy'):
        pya.MessageBox.warning(
        "Missing package", "The OPICS circuit simulator does not function without the package 'scipy'.",  pya.MessageBox.Ok)    
        return None
    
    
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
    
    '''
    import numpy as np
    from SiEPIC.opics.globals import C
    freq = np.linspace(C * 1e6 / 1.5, C * 1e6 / 1.6, 2000)
    circuit = Network(network_id="circuit_name", f=freq)
    '''
    
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
    wavelengths = c_/subckt.f

    # collect all the results for each port:
    import numpy as np
    nports = subckt.sim_result.nports
    out = result['S_1_0']
    print(out.shape)       
    columns = ['Output 1']
    for i in range(1,nports-1):
        print(out.shape)       
        out = np.vstack((out, result['S_%s_0' %(i+1)]))
        columns = columns + ['Output %s' % (i+1) ]
    
    # Single line:
    #df = pd.DataFrame(transmission, index=wavelengths, columns=['Transmission'])
    
    # Two lines:
    df = pd.DataFrame(out.transpose(), index=wavelengths, columns=columns)
    fig = px.line(df, labels={'index':'Wavelength', 'value':'Transmission (dB)'}, markers=True)
    fig.show()
    
    
