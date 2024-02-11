"""
This file is part of the SiEPIC-Tools and SiEPIC-GSiP PDK
by Lukas Chrostowski (c) 2015-2017

This Python file implements a library called "GSiP" for scripted and GUI-based layout flows. 

Crash warning:
 https://www.klayout.de/forum/comments.php?DiscussionID=734&amp;page=1#Item_13
 This library has nested PCells. Running this macro with a layout open may
 cause it to crash. Close the layout first before running.

Version history:

Mustafa Hammood 2020/6/25
- Refactored PCells out of library files into individual files in a subdirectory

Jaspreet Jhoja 2020/5/23
- Refactored PCells to make them compatible with both, GUI and script-based layout operations

Stefan Preble and Karl McNulty (RIT) 2019/6/13
 - Wireguide : Path to metal wires 

Lukas Chrostowski 2017/12/16
 - compatibility with KLayout 0.25 and SiEPIC-Tools

Lukas Chrostowski
 - GDS cells (detector, etc) and PCells (ring modulator, filter)
 
todo:
replace:     
 layout_arc_wg_dbu(self.cell, Layerm1N, x0,y0, r_m1_in, w_m1_in, angle_min_doping, angle_max_doping)
with:
 self.cell.shapes(Layerm1N).insert(pya.Polygon(arc(w_m1_in, angle_min_doping, angle_max_doping) + [pya.Point(0, 0)]).transformed(t))
"""

folder = 'pcells_GSiP'
verbose = False

import os, sys, pathlib

dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

try:
    import SiEPIC
except:
    dir_path_SiEPIC = os.path.join(dir_path, '../../../python')
    sys.path.append(dir_path_SiEPIC)
    import SiEPIC

from SiEPIC._globals import KLAYOUT_VERSION, KLAYOUT_VERSION_3
if KLAYOUT_VERSION < 28:
    question = pya.QMessageBox()
    question.setStandardButtons(pya.QMessageBox.Ok)
    question.setText("This PDK is not compatible with older versions (<0.28) of KLayout.")
    KLayout_link0='https://www.klayout.de/build.html'
    question.setInformativeText("\nThis PDK is not compatible with older versions (<0.28) of KLayout.\nPlease download an install the latest version, from %s" % (KLayout_link0))
    pya.QMessageBox_StandardButton(question.exec_())

files = [f for f in os.listdir(os.path.join(os.path.dirname(
    os.path.realpath(__file__)),folder)) if '.py' in pathlib.Path(f).suffixes  and '__init__' not in f]
import importlib
pcells_GSiP = importlib.import_module(folder) 
importlib.invalidate_caches()
pcells_=[]
for f in files:
    module = '%s.%s' % (folder, f.replace('.py',''))  ### folder name ###
    if verbose:
        print(' - found module: %s' % module)
    m = importlib.import_module(module) 
    if verbose:
        print(m)
    pcells_.append(importlib.reload(m))

import pya

class GSiP(pya.Library):
  def __init__(self):
    tech_name = "GSiP"
    library = tech_name
    self.technology=tech_name


    if verbose:
        print("Initializing '%s' Library." % library)

    # Set the description
    self.description = "SiEPIC Generic SiP"

    # Save the path, used for loading WAVEGUIDES.XML
    import os
    self.path = os.path.dirname(os.path.realpath(__file__))

    # Import all the GDS files from the tech folder
    import os, fnmatch
    dir_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gds/building_blocks"))
    if verbose:
        print('  library path: %s' % dir_path)
    search_str = '*.[Oo][Aa][Ss]' # OAS
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, search_str):
            file1=os.path.join(root, filename)
            if verbose:
                print(" - reading %s" % file1 )
            self.layout().read(file1)
    search_str = '*.[Gg][Dd][Ss]' # GDS
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, search_str):
            file1=os.path.join(root, filename)
            if verbose:
                print(" - reading %s" % file1 )
            self.layout().read(file1)
                        
    # Create the PCell declarations
    for m in pcells_:
        mm = m.__name__.replace('%s.' % folder,'')
        mm2 = m.__name__+'.'+mm+'()'
        if verbose:
            print(' - register_pcell %s, %s' % (mm,mm2))
        self.layout().register_pcell(mm, eval(mm2))
                
    if verbose:
        print(' done with pcells')
    
    # Register us the library with the technology name
    # If a library with that name already existed, it will be replaced then.
    self.register(library)
     
  
GSiP()
