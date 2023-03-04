#%% append Python path to code location
import os,sys,inspect

# change directory for database
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 
os.chdir(parent_dir) 

# import ContraDC module
from ContraDC import *


def examples(num):
    """ Function implements 4 use-case examples """

    """ Example 1: regular SOI Contra-DC """
    if num ==1:

        # instantiate, simulate and show result
        device = ContraDC().simulate().displayResults()

        # calculate thimpe group delay
        device.getGroupDelay()

        # plot group delay
        plt.figure()
        plt.plot(device.wavelength*1e9, device.group_delay*1e12)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Tg (ps)")

        plt.show()



    """ Example 2: Full chirped example.
        Create a CDC with chirped w1, w2, period, temperature.
    """
    if num == 2:
    
        # Waveguide chirp
        w1 = [.56e-6, .56e-6]
        w2 = [.44e-6, .44e-6]
        w_chirp_step = .1e-9

        # Period chirp
        period = [310e-9, 320e-9]
        

        # apod shape
        apod_shape = "gaussian"

        N = 10000

        device = ContraDC(N=N, w1=w1, w2=w2, apod_shape=apod_shape, period_chirp_step=1e-9,
                         w_chirp_step=w_chirp_step, period=period, N_seg=1500,
                         kappa = 10000, a=0, alpha=1.5, wvl_range=[1500e-9,1600e-9])

        device.simulate().displayResults()



    """ Example 3: defining custom chirp profiles
    """
    if num == 3:

        device = ContraDC(apod_shape="tanh")

        z = np.linspace(0, device.N_seg, device.N_seg)
        device.w1_profile = device.w1*np.cos(z/600)
        device.w2_profile = device.w2*np.cos(z/600)

        device.simulate().displayResults()



    """ Example 4: using custom supermode indices.
        You might want to use this if you are designing 
        with silicon nitride, of using other waveguide specs than
        SOI, 100-nm gap.
    """
    if num == 4:

        device = ContraDC(polyfit_file="Tutorial/SiN_1550_TE_w1_850nm_w2_1150nm_thickness_400nm.txt", period=335e-9)
        device.simulate().displayResults()



    """Example 5: Lumerical-assisted flow
    """
    if num == 5:

        apod_shape = "gaussian"
        period = 316e-9
        w1 = 560e-9
        w2 = 440e-9

        device = ContraDC(w1= w1, w2=w2, apod_shape=apod_shape, period=period, kappa=25000)

        device.simulate()
        import plotly.graph_objs as go
        import plotly.offline as pyo
        drop = go.Scatter(x=device.wavelength*1e9, y=device.drop, mode='lines', name='Through')
        thru = go.Scatter(x=device.wavelength*1e9, y=device.thru, mode='lines', name='Drop')
        layout = go.Layout(title='Contra-directional coupler device', xaxis=dict(title='X Axis'), yaxis=dict(title='Y Axis'))
        fig = go.Figure(data=[thru, drop], layout=layout)
        fig.show()
        print('aaaa')


        # Generate compact model for Lumerical INTERCONNECT
        #device.gen_sparams() # this will create a ContraDC_sparams.dat file to import into INTC

    """Example 6: Complete Lumerical flow - simulate coupling coefficient (not simulating mode profiles)
    """
    if num == 6:

        apod_shape = "tanh"
        period = 318e-9
        w1 = 560e-9
        dw1 = 25e-9
        w2 = 440e-9
        dw2 = 50e-9
        gap = 100e-9

        device = ContraDC(w1= w1, dw1=dw1, w2=w2, dw2=dw2, gap=gap, apod_shape=apod_shape, period=period)

        device.simulate_kappa()
        device.simulate().displayResults()
        plt.plot(device.wavelength*1e9, device.drop)
        plt.plot(device.wavelength*1e9, device.thru)
        plt.show()

        # Generate compact model for Lumerical INTERCONNECT
        #device.gen_sparams() # this will create a ContraDC_sparams.dat file to import into INTC


examples(5)
