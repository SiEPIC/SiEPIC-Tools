
# Required packages
from SiEPIC.install import install
if not install('scipy', requested_by='Contra Directional Coupler design'):
  pya.MessageBox.warning(
  "Missing package", "The simulator does not function without the package 'scipy'.",  pya.MessageBox.Ok)    
if not install('plotly', requested_by='Contra Directional Coupler design'):
  pya.MessageBox.warning(
  "Missing package", "The simulator does not function without the package 'plotly'.",  pya.MessageBox.Ok)    



from SiEPIC.simulation.contraDC.contra_directional_coupler.ContraDC import *
import sys
import pya
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
pio.renderers.default = "browser"

class MyWindow(pya.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Contra-directional coupler simulator')
        self.setMinimumSize(200, 200)

        # fetch pcell parameters
        if self.load_pcell_params() == 0:
            self.close()
        
        # fetch technology parameters
        self.load_DFT()

        #******************************************************
        # Create the layout_pcell and add the UI elements to it
        from pya import QVBoxLayout, QPushButton, QLabel, QLineEdit, QCheckBox, QComboBox, QHBoxLayout
        layout_pcell = QVBoxLayout()
        self.button = QPushButton('Refresh PCell')
        # Connect the button to a callback function
        self.button.clicked(self.on_refresh_clicked)

        self.label_pcell = QLabel('Parameterized cell definitions:')
        layout_pcell.addWidget(self.label_pcell)

        self.pcell_N_label = QLabel('Number of gratings (N) (µm): ')
        self.pcell_N_fill = QLineEdit(str(self.params['number_of_periods']))
        self.pcell_N_fill.setReadOnly(True)
        self.pcell_N_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_N_label)
        layout_pcell.addWidget(self.pcell_N_fill)

        self.pcell_period_label = QLabel('Gratings period (Λ) (µm): ')
        self.pcell_period_fill = QLineEdit(str(self.params['grating_period']))
        self.pcell_period_fill.setReadOnly(True)
        self.pcell_period_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_period_label)
        layout_pcell.addWidget(self.pcell_period_fill)

        self.pcell_gap_label = QLabel('Waveguides gap (G) (µm): ')
        self.pcell_gap_fill = QLineEdit(str(self.params['gap']))
        self.pcell_gap_fill.setReadOnly(True)
        self.pcell_gap_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_gap_label)
        layout_pcell.addWidget(self.pcell_gap_fill)

        self.pcell_w1_label = QLabel('Waveguide 1 width (W1) (µm): ')
        self.pcell_w1_fill = QLineEdit(str(self.params['wg1_width']))
        self.pcell_w1_fill.setReadOnly(True)
        self.pcell_w1_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_w1_label)
        layout_pcell.addWidget(self.pcell_w1_fill)

        self.pcell_dw1_label = QLabel('Waveguide 1 Δwidth (ΔW1) (µm): ')
        self.pcell_dw1_fill = QLineEdit(str(self.params['corrugation1_width']))
        self.pcell_dw1_fill.setReadOnly(True)
        self.pcell_dw1_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_dw1_label)
        layout_pcell.addWidget(self.pcell_dw1_fill)

        self.pcell_w2_label = QLabel('Waveguide 2 width (W2) (µm): ')
        self.pcell_w2_fill = QLineEdit(str(self.params['wg2_width']))
        self.pcell_w2_fill.setReadOnly(True)
        self.pcell_w2_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_w2_label)
        layout_pcell.addWidget(self.pcell_w2_fill)

        self.pcell_dw2_label = QLabel('Waveguide 2 Δwidth (ΔW2) (µm): ')
        self.pcell_dw2_fill = QLineEdit(str(self.params['corrugation2_width']))
        self.pcell_dw2_fill.setReadOnly(True)
        self.pcell_dw2_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_dw2_label)
        layout_pcell.addWidget(self.pcell_dw2_fill)

        self.pcell_apod_label = QLabel('Apodization index (a): ')
        self.pcell_apod_fill = QLineEdit(str(self.params['apodization_index']))
        self.pcell_apod_fill.setReadOnly(True)
        self.pcell_apod_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_apod_label)
        layout_pcell.addWidget(self.pcell_apod_fill)

        self.pcell_rib_label = QLabel('Rib waveguides? ')
        self.pcell_rib_fill = QCheckBox()
        self.pcell_rib_fill.setChecked(self.params['rib'])
        layout_pcell.addWidget(self.pcell_rib_label)
        layout_pcell.addWidget(self.pcell_rib_fill)
        self.pcell_rib_fill.clicked(self.on_rib_click)

        layout_pcell.addWidget(self.button)

        #******************************************************
        # add simulation box and add the UI elements to it
        layout_sim = QVBoxLayout()
        self.label_sim = QLabel('Simulation definitions:')
        layout_sim.addWidget(self.label_sim)

        # Create a dropdown menu to select simulation import type
        self.sim_import_label = QLabel('Import simulation definitions from:')
        self.sim_import = QComboBox()
        self.sim_import.addItem("PDK definitions")
        self.sim_import.addItem("Custom")
        layout_sim.addWidget(self.sim_import_label)
        layout_sim.addWidget(self.sim_import)

        # Connect the dropdown menu to a slot function
        self.sim_import.currentIndexChanged(self.on_sim_import)

        self.sim_wavlstart_label = QLabel('Start wavelength (µm): ')
        self.sim_wavlstart_fill = QLineEdit(str(self.wavl_start))
        self.sim_wavlstart_fill.setReadOnly(True)
        self.sim_wavlstart_fill.setStyleSheet('color: gray')
        layout_sim.addWidget(self.sim_wavlstart_label)
        layout_sim.addWidget(self.sim_wavlstart_fill)

        self.sim_wavlstop_label = QLabel('Stop wavelength (µm): ')
        self.sim_wavlstop_fill = QLineEdit(str(self.wavl_stop))
        self.sim_wavlstop_fill.setReadOnly(True)
        self.sim_wavlstop_fill.setStyleSheet('color: gray')
        layout_sim.addWidget(self.sim_wavlstop_label)
        layout_sim.addWidget(self.sim_wavlstop_fill)

        self.sim_wavlpts_label = QLabel('Wavelength points: ')
        self.sim_wavlpts_fill = QLineEdit(str(self.wavl_pts))
        self.sim_wavlpts_fill.setReadOnly(True)
        self.sim_wavlpts_fill.setStyleSheet('color: gray')
        layout_sim.addWidget(self.sim_wavlpts_label)
        layout_sim.addWidget(self.sim_wavlpts_fill)

        # Polarization
        self.sim_pol_label = QLabel('Polarization: ')
        self.sim_pol_dropdown = QComboBox()
        self.sim_pol_dropdown.addItem("TE")
        self.sim_pol_dropdown.addItem("TM")
        layout_sim.addWidget(self.sim_pol_label)
        layout_sim.addWidget(self.sim_pol_dropdown)

        # coupling coefficient
        self.sim_kappa_label = QLabel('Coupling coefficient (κ, /m): ')
        self.sim_kappa_dropdown = QComboBox()
        self.sim_kappa_dropdown.addItem("User defined")
        self.sim_kappa_dropdown.addItem("Simulate")
        self.sim_kappa_fill = QLineEdit('24000')
        self.sim_kappa_fill.setReadOnly(True)
        self.sim_kappa_fill.setStyleSheet('color: gray')
        layout_sim.addWidget(self.sim_kappa_label)
        layout_sim.addWidget(self.sim_kappa_dropdown)
        layout_sim.addWidget(self.sim_kappa_fill)

        # Connect the dropdown menu to a slot function
        self.sim_kappa_dropdown.currentIndexChanged(self.on_kappa_dropdown)

        # waveguide models
        self.sim_wg_label = QLabel('Waveguide models: ')
        self.sim_wg_dropdown = QComboBox()
        self.sim_wg_dropdown.addItem("Lookup table")
        self.sim_wg_dropdown.addItem("Simulate")
        layout_sim.addWidget(self.sim_wg_label)
        layout_sim.addWidget(self.sim_wg_dropdown)

        # Connect the dropdown menu to a slot function
        self.sim_wg_dropdown.currentIndexChanged(self.on_wg_dropdown)

        #******************************************************
        # add technology box and add the UI elements to it
        layout_tech = QVBoxLayout()
        
        self.label_tech = QLabel('Technology definitions:')
        layout_tech.addWidget(self.label_tech)

        # Create a dropdown menu to select simulation import type
        self.tech_import_label = QLabel('Import techonology definitions from: ')
        self.tech_import = QComboBox()
        self.tech_import.addItem("PDK definitions")
        self.tech_import.addItem("Custom")
        layout_tech.addWidget(self.tech_import_label)
        layout_tech.addWidget(self.tech_import)

        self.tech_devthick_label = QLabel('Waveguide thickness (µm): ')
        self.tech_devthick_fill = QLineEdit('0.22')
        self.tech_devthick_fill.setReadOnly(False)
        layout_tech.addWidget(self.tech_devthick_label)
        layout_tech.addWidget(self.tech_devthick_fill)

        self.tech_ribthick_label = QLabel('Rib thickness (µm): ')
        if self.pcell_rib_fill.isChecked():
          self.tech_ribthick_fill = QLineEdit('0.09')
          self.tech_ribthick_fill.setReadOnly(False)
        else:
          self.tech_ribthick_fill = QLineEdit('0.0')
          self.tech_ribthick_fill.setReadOnly(True)
          self.tech_ribthick_fill.setStyleSheet('color: gray')
        layout_tech.addWidget(self.tech_ribthick_label)
        layout_tech.addWidget(self.tech_ribthick_fill)

        self.tech_plot_label = QLabel('Plot result? ')
        self.tech_plot_fill = QCheckBox()
        self.tech_plot_fill.setChecked(True)
        layout_tech.addWidget(self.tech_plot_label)
        layout_tech.addWidget(self.tech_plot_fill)

        self.tech_cm_label = QLabel('Generate compact model? ')
        self.tech_cm_fill = QCheckBox()
        self.tech_cm_fill.setChecked(True)
        layout_tech.addWidget(self.tech_cm_label)
        layout_tech.addWidget(self.tech_cm_fill)

        self.simulate = QPushButton('Run simulation')
        layout_tech.addWidget(self.simulate)

        # Connect the button to a callback function
        self.simulate.clicked(self.on_simulate_clicked)

        #******************************************************
        # assemble and order the menus
        layout_pcell.addStretch()
        layout_sim.addStretch()
        layout_tech.addStretch()
        hbox = QHBoxLayout(self)
        hbox.addLayout(layout_pcell)
        hbox.addSpacing(20)
        hbox.addLayout(layout_sim)
        hbox.addSpacing(20)
        hbox.addLayout(layout_tech)

        vbox = QVBoxLayout(self)
        vbox.addLayout(hbox)

        self.setLayout(vbox)


    def on_simulate_clicked(self):
        
        

        N = float(self.pcell_N_fill.text)        
        period = float(self.pcell_period_fill.text)*1e-6
        gap = float(self.pcell_gap_fill.text)*1e-6
        w1 = float(self.pcell_w1_fill.text)*1e-6
        w2 = float(self.pcell_w2_fill.text)*1e-6
        dw1 = float(self.pcell_dw1_fill.text)*1e-6
        dw2 = float(self.pcell_dw2_fill.text)*1e-6
        a = float(self.pcell_apod_fill.text)
        if self.sim_pol_dropdown.currentIndex == 0:
            pol = 'TE'
        else:
            pol = 'TM'
        if self.pcell_rib_fill.isChecked():
            rib = True
        else:
            rib = False

        thickness_device = float(self.tech_devthick_fill.text)*1e-6
        thickness_rib = float(self.tech_ribthick_fill.text)*1e-6
        wvl_range = [float(self.sim_wavlstart_fill.text)*1e-9, float(self.sim_wavlstop_fill.text)*1e-9]

        device = ContraDC(w1= w1, dw1=dw1, w2=w2, dw2=dw2, gap=gap, a=a, period=period, rib=rib,
        pol=pol, thickness_device=thickness_device, thickness_rib=thickness_rib, wvl_range=wvl_range)

        if self.sim_kappa_dropdown.currentIndex == 0:
            device.kappa = float(self.sim_kappa_fill.text)
        else:
            device.simulate_kappa()

        self.simulate.setText('Simulating...')
        device.simulate()

        if self.tech_plot_fill.isChecked():
            import plotly.graph_objs as go
            import plotly.offline as pyo
            import plotly.io as pio
            #pio.renderers.default = "browser"
            
            drop = go.Scatter(x=device.wavelength*1e9, y=device.drop, mode='lines', name='Through')
            thru = go.Scatter(x=device.wavelength*1e9, y=device.thru, mode='lines', name='Drop')
            layout = go.Layout(title='Contra-directional coupler device', xaxis=dict(title='X Axis'), yaxis=dict(title='Y Axis'))
            fig = go.Figure(data=[thru, drop], layout=layout)
            fig.show()
        
        if self.tech_cm_fill.isChecked():
            # Check if there is a layout open, so we know which technology to install
            lv = pya.Application.instance().main_window().current_view()
            if lv == None:
                raise UserWarning("To save data to the Compact Model Library, first, please create a new layout and select the desired technology:\n  Menu: File > New Layout, and a Technology.\nThen repeat.")
            
            # Get the Technology 
            from SiEPIC.utils import get_layout_variables
            TECHNOLOGY, lv, ly, top_cell = get_layout_variables()
            
            # Check if there is a CML folder in the Technology folder
            import os
            base_path = ly.technology().base_path()    
            folder_CML = os.path.join(base_path,'CML/%s/source_data/contraDC' % ly.technology().name)
            if not os.path.exists(folder_CML):
                raise UserWarning("The folder %s does not exist. \nCannot save to the Compact Model Library." %folder_CML)
            
            # Generate compact model for Lumerical INTERCONNECT
            # return self.path_dat, .dat file that was created
            device.gen_sparams(filepath=folder_CML, make_plot=False) # this will create a ContraDC_sparams.dat file to import into INTC


        self.device = device
        self.simulate.setText('Done simulating.')


    def on_refresh_clicked(self):
        # fetch pcell parameters
        if self.load_pcell_params() == 0:
            return
        self.pcell_N_fill.setText(str(self.params['number_of_periods']))
        self.pcell_period_fill.setText(str(self.params['grating_period']))
        self.pcell_gap_fill.setText(str(self.params['gap']))
        self.pcell_w1_fill.setText(str(self.params['wg1_width']))
        self.pcell_dw1_fill.setText(str(self.params['corrugation1_width']))
        self.pcell_w2_fill.setText(str(self.params['wg2_width']))
        self.pcell_dw2_fill.setText(str(self.params['corrugation2_width']))
        self.pcell_apod_fill.setText(str(self.params['apodization_index']))
        self.pcell_rib_fill.setChecked(self.params['rib'])

        self.label_pcell.setText('Parameterized cell refreshed...')
        self.simulate.setText('Simulate')

    def on_sim_import(self):
        if self.sim_import.currentIndex == 0:
            # import simulation parameters from DFT
            self.load_DFT()
            self.sim_import_label.setText('Simulation definitions: PDK')
            self.sim_wavlstart_fill.setText(str(self.wavl_start))
            self.sim_wavlstart_fill.setReadOnly(True)
            self.sim_wavlstart_fill.setStyleSheet('color: gray')

            self.sim_wavlstop_fill.setText(str(self.wavl_stop))
            self.sim_wavlstop_fill.setReadOnly(True)
            self.sim_wavlstop_fill.setStyleSheet('color: gray')

            self.sim_wavlpts_fill.setText(str(self.wavl_pts))
            self.sim_wavlpts_fill.setReadOnly(True)
            self.sim_wavlpts_fill.setStyleSheet('color: gray')
        else:
            # let user pick and ungrey the boxes
            self.sim_import_label.setText('Simulation definitions: Custom')
            self.sim_wavlstart_fill.setReadOnly(False)
            self.sim_wavlstart_fill.setStyleSheet('color: black')

            self.sim_wavlstop_fill.setReadOnly(False)
            self.sim_wavlstop_fill.setStyleSheet('color: black')

            self.sim_wavlpts_fill.setReadOnly(False)
            self.sim_wavlpts_fill.setStyleSheet('color: black')

    def on_kappa_dropdown(self):
        if self.sim_kappa_dropdown.currentIndex == 0:
            # query user for input
            self.sim_kappa_label.setText('Coupling coefficient (κ): Custom')
            self.sim_kappa_fill.setReadOnly(False)
            self.sim_kappa_fill.setStyleSheet('color: black')
            self.sim_kappa_fill.setText('Kappa (/m)')
        else:
            # simulate kappa using Lumerical
            self.sim_kappa_label.setText('Coupling coefficient (κ): Simulate')
            self.sim_kappa_fill.setReadOnly(True)
            self.sim_kappa_fill.setStyleSheet('color: gray')
            self.sim_kappa_fill.setText('simulation')

    def on_wg_dropdown(self):
        if self.sim_wg_dropdown.currentIndex == 0:
            # look up if value is within lookup table index
            self.sim_wg_label.setText('Waveguide models: LUT')
        else:
            # simulate modes using Lumerical
            self.sim_wg_label.setText('Waveguide models: Simulate')

    def on_rib_click(self):
        if self.pcell_rib_fill.isChecked():
            self.tech_ribthick_fill.setText('rib thickness (µm)')
            self.tech_ribthick_fill.setStyleSheet('color: black')
            self.tech_ribthick_fill.setReadOnly(False)
        else:
            self.tech_ribthick_fill.setText('0 nm')
            self.tech_ribthick_fill.setReadOnly(True)
            self.tech_ribthick_fill.setStyleSheet('color: gray')

    def load_DFT(self):
        from SiEPIC.utils import load_DFT
        DFT = load_DFT()
        self.wavl_start = DFT['design-for-test']['tunable-laser'][0]['wavelength-start']
        self.wavl_stop = DFT['design-for-test']['tunable-laser'][0]['wavelength-stop']
        self.wavl_pts = DFT['design-for-test']['tunable-laser'][0]['wavelength-points']
        self.pol = DFT['design-for-test']['tunable-laser'][0]['polarization']

    def load_pcell_params(self):
        # get selected instances; only one
        from SiEPIC.utils import select_instances, get_layout_variables
        TECHNOLOGY, lv, ly, cell = get_layout_variables()
    
        # print error message if no or more than one component selected
        selected_instances = select_instances()
        error = pya.QMessageBox()
        error.setStandardButtons(pya.QMessageBox.Ok)
        if len(selected_instances) != 1:
            error.setText("Error: Need to have one component selected.")
            response = error.exec_()
            return 0
    
        for obj in selected_instances:
            c = cell.find_components(cell_selected=[obj.inst().cell], verbose=True)
    
        # check if selected PCell is a contra DC
        if c[0].cell.basic_name() != "contra_directional_coupler":
            error.setText("Error: selected component must be a contra_directional_coupler PCell.")
            response = error.exec_()
            return 0
    
        # parse PCell parameters into params array
        if c[0].cell.is_pcell_variant():
            self.params = c[0].cell.pcell_parameters_by_name()
        else:
            error.setText("Error: selected component must be a contra-DC PCell.")
            response = error.exec_()
            return 0

    # Define the exit function
    def exit(self):
        self.close()

def cdc_gui():
    app = pya.QApplication.instance()
    if app is None:
        app = pya.QApplication([])

#    import SiEPIC._globals
#    SiEPIC._globals.GUI_cdc = MyWindow()
#    print(SiEPIC._globals.GUI_cdc)
#    SiEPIC._globals.GUI_cdc.show()
    GUI_cdc.show()
    
    app.exec_()

GUI_cdc = MyWindow()
print('CDC Gui: %s' % GUI_cdc)
