# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 23:19:50 2023

@author: mustafa
"""

import sys
import pya


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Contra-directional coupler simulator')
        self.setMinimumSize(400, 300)

        # fetch pcell parameters
        # fetch technology parameters

        # Create the layout_pcell and add the UI elements to it
        layout_pcell = QVBoxLayout()
        self.button = QPushButton('Refresh PCell')
        # Connect the button to a callback function
        self.button.clicked(self.on_refresh_clicked)

        self.label_pcell = QLabel('Parameterized cell definitions:')
        layout_pcell.addWidget(self.label_pcell)

        self.pcell_N_label = QLabel('Number of gratings (N): ')
        self.pcell_N_fill = QLineEdit('?')
        self.pcell_N_fill.setReadOnly(True)
        self.pcell_N_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_N_label)
        layout_pcell.addWidget(self.pcell_N_fill)

        self.pcell_period_label = QLabel('Gratings period (Λ): ')
        self.pcell_period_fill = QLineEdit('?')
        self.pcell_period_fill.setReadOnly(True)
        self.pcell_period_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_period_label)
        layout_pcell.addWidget(self.pcell_period_fill)

        self.pcell_gap_label = QLabel('Waveguides gap (G): ')
        self.pcell_gap_fill = QLineEdit('?')
        self.pcell_gap_fill.setReadOnly(True)
        self.pcell_gap_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_gap_label)
        layout_pcell.addWidget(self.pcell_gap_fill)

        self.pcell_w1_label = QLabel('Waveguide 1 width (W1): ')
        self.pcell_w1_fill = QLineEdit('?')
        self.pcell_w1_fill.setReadOnly(True)
        self.pcell_w1_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_w1_label)
        layout_pcell.addWidget(self.pcell_w1_fill)

        self.pcell_dw1_label = QLabel('Waveguide 1 Δwidth (ΔW1): ')
        self.pcell_dw1_fill = QLineEdit('?')
        self.pcell_dw1_fill.setReadOnly(True)
        self.pcell_dw1_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_dw1_label)
        layout_pcell.addWidget(self.pcell_dw1_fill)

        self.pcell_w2_label = QLabel('Waveguide 2 width (W2): ')
        self.pcell_w2_fill = QLineEdit('?')
        self.pcell_w2_fill.setReadOnly(True)
        self.pcell_w2_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_w2_label)
        layout_pcell.addWidget(self.pcell_w2_fill)

        self.pcell_dw2_label = QLabel('Waveguide 2 Δwidth (ΔW2): ')
        self.pcell_dw2_fill = QLineEdit('?')
        self.pcell_dw2_fill.setReadOnly(True)
        self.pcell_dw2_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_dw2_label)
        layout_pcell.addWidget(self.pcell_dw2_fill)

        self.pcell_apod_label = QLabel('Apodization index (a): ')
        self.pcell_apod_fill = QLineEdit('?')
        self.pcell_apod_fill.setReadOnly(True)
        self.pcell_apod_fill.setStyleSheet('color: gray')
        layout_pcell.addWidget(self.pcell_apod_label)
        layout_pcell.addWidget(self.pcell_apod_fill)

        self.pcell_rib_label = QLabel('Rib waveguides? ')
        self.pcell_rib_fill = QCheckBox()
        self.pcell_rib_fill.setChecked(True)
        self.pcell_rib_fill.setDisabled(True)
        layout_pcell.addWidget(self.pcell_rib_label)
        layout_pcell.addWidget(self.pcell_rib_fill)

        layout_pcell.addWidget(self.button)
        layout_pcell.addStretch()

        # add technology box and add the UI elements to it
        layout_tech = QVBoxLayout()
        self.label_tech = QLabel('Simulation definitions:')
        layout_tech.addWidget(self.label_tech)

        # Create a dropdown menu to select simulation import type
        self.tech_import_label = QLabel('Import simulation definitions from:')
        self.tech_import = QComboBox()
        self.tech_import.addItem("PDK definitions")
        self.tech_import.addItem("Custom")
        layout_tech.addWidget(self.tech_import_label)
        layout_tech.addWidget(self.tech_import)

        # Connect the dropdown menu to a slot function
        self.tech_import.currentIndexChanged(self.on_tech_import)

        self.tech_wavlstart_label = QLabel('Start wavelength: ')
        self.tech_wavlstart_fill = QLineEdit('?')
        self.tech_wavlstart_fill.setReadOnly(True)
        self.tech_wavlstart_fill.setStyleSheet('color: gray')
        layout_tech.addWidget(self.tech_wavlstart_label)
        layout_tech.addWidget(self.tech_wavlstart_fill)

        self.tech_wavlstop_label = QLabel('Stop wavelength: ')
        self.tech_wavlstop_fill = QLineEdit('?')
        self.tech_wavlstop_fill.setReadOnly(True)
        self.tech_wavlstop_fill.setStyleSheet('color: gray')
        layout_tech.addWidget(self.tech_wavlstop_label)
        layout_tech.addWidget(self.tech_wavlstop_fill)

        self.tech_wavlpts_label = QLabel('Wavelength points: ')
        self.tech_wavlpts_fill = QLineEdit('?')
        self.tech_wavlpts_fill.setReadOnly(True)
        self.tech_wavlpts_fill.setStyleSheet('color: gray')
        layout_tech.addWidget(self.tech_wavlpts_label)
        layout_tech.addWidget(self.tech_wavlpts_fill)

        # Polarization
        self.tech_pol_label = QLabel('Polarization: ')
        self.tech_pol_dropdown = QComboBox()
        self.tech_pol_dropdown.addItem("TE")
        self.tech_pol_dropdown.addItem("TM")
        layout_tech.addWidget(self.tech_pol_label)
        layout_tech.addWidget(self.tech_pol_dropdown)

        # coupling coefficient
        self.tech_kappa_label = QLabel('Coupling coefficient (κ): ')
        self.tech_kappa_dropdown = QComboBox()
        self.tech_kappa_dropdown.addItem("User defined")
        self.tech_kappa_dropdown.addItem("Simulate")
        self.tech_kappa_fill = QLineEdit('?')
        self.tech_kappa_fill.setReadOnly(True)
        self.tech_kappa_fill.setStyleSheet('color: gray')
        layout_tech.addWidget(self.tech_kappa_label)
        layout_tech.addWidget(self.tech_kappa_dropdown)
        layout_tech.addWidget(self.tech_kappa_fill)

        # Connect the dropdown menu to a slot function
        self.tech_kappa_dropdown.currentIndexChanged(self.on_kappa_dropdown)

        # waveguide models
        self.tech_wg_label = QLabel('Waveguide models: ')
        self.tech_wg_dropdown = QComboBox()
        self.tech_wg_dropdown.addItem("Lookup table")
        self.tech_wg_dropdown.addItem("Simulate")
        layout_tech.addWidget(self.tech_wg_label)
        layout_tech.addWidget(self.tech_wg_dropdown)

        # Connect the dropdown menu to a slot function
        self.tech_wg_dropdown.currentIndexChanged(self.on_wg_dropdown)

        hbox = QHBoxLayout(self)
        hbox.addLayout(layout_pcell)
        hbox.addLayout(layout_tech)
        self.setLayout(hbox)

    def on_refresh_clicked(self):
        self.label_pcell.setText('Parameterized cell definitions: refreshed.')

    def on_tech_import(self):
        if self.tech_import.currentIndex == 0:
            # import simulation parameters from DFT
            self.tech_import_label.setText('Simulation definitions: PDK')
        else:
            # let user pick and ungrey the boxes
            self.tech_import_label.setText('Simulation definitions: Custom')

    def on_kappa_dropdown(self):
        if self.tech_kappa_dropdown.currentIndex == 0:
            # query user for input
            self.tech_kappa_label.setText('Coupling coefficient (κ): Custom')
            self.tech_kappa_fill.setReadOnly(False)
            self.tech_kappa_fill.setStyleSheet('color: black')
            self.tech_kappa_fill.setText('Kappa (/m)')
        else:
            # simulate kappa using Lumerical
            self.tech_kappa_label.setText('Coupling coefficient (κ): Simulate')
            self.tech_kappa_fill.setReadOnly(False)
            self.tech_kappa_fill.setStyleSheet('color: gray')
            self.tech_kappa_fill.setText('simulation')

    def on_wg_dropdown(self):
        if self.tech_wg_dropdown.currentIndex == 0:
            # look up if value is within lookup table index
            self.tech_wg_label.setText('Waveguide models: LUT')
        else:
            # simulate modes using Lumerical
            self.tech_wg_label.setText('Waveguide models: Simulate')


# Create the application and the main window
app = pya.QApplication.instance()
if app is None:
    app = pya.QApplication([])
mw = MyWindow()

# Show the main window and run the application
mw.show()
app.exec_()


def contraDC_menu(verbose=True):
    import pya
    # get selected instances; only one
    from SiEPIC.utils import select_instances, get_layout_variables, load_DFT
    from SiEPIC import _globals

    DFT = load_DFT()
    wavl_start = DFT['design-for-test']['tunable-laser'][0]['wavelength-start']
    wavl_stop = DFT['design-for-test']['tunable-laser'][0]['wavelength-stop']
    wavl_pts = DFT['design-for-test']['tunable-laser'][0]['wavelength-points']
    pol = DFT['design-for-test']['tunable-laser'][0]['polarization']

    TECHNOLOGY, lv, ly, cell = get_layout_variables()

    # print error message if no or more than one component selected
    selected_instances = select_instances()
    error = pya.QMessageBox()
    error.setStandardButtons(pya.QMessageBox.Ok)
    if len(selected_instances) != 1:
        error.setText("Error: Need to have one component selected.")
        response = error.exec_()
        return

    for obj in selected_instances:
        c = cell.find_components(cell_selected=[obj.inst().cell], verbose=True)

    # check if selected PCell is a contra DC
    if c[0].cell.basic_name() != "contra_directional_coupler":
        error.setText("Error: selected component must be a contra_directional_coupler PCell.")
        response = error.exec_()
        return

    # parse PCell parameters into params array
    if c[0].cell.is_pcell_variant():
        params = c[0].cell.pcell_parameters_by_name()
    else:
        error.setText("Error: selected component must be a contra-DC PCell.")
        response = error.exec_()
        return
    print(params)
