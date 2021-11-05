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

import os
import pcells_GSiP
import pya

class GSiP(pya.Library):
  def __init__(self):
    tech_name = "GSiP"
    library = tech_name
    self.technology=tech_name
    
    print("Initializing '%s' Library." % library)

    self.description = "SiEPIC Generic SiP"

    # Save the path, used for loading WAVEGUIDES.XML
    import os
    self.path = os.path.dirname(os.path.realpath(__file__))

    # Import all the GDS files from the tech folder "gds"
    import os, fnmatch
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gds/building_blocks")
    search_str = '*' + '.gds'
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter([f.lower() for f in filenames], search_str):
            file1=os.path.join(root, filename)
            print(" - reading %s" % file1 )
            self.layout().read(file1)

    for attr, value in pcells_GSiP.__dict__.items():
      try:
        if value.__module__.split('.')[0] == 'pcells_GSiP':
          print('Registered pcell: '+attr)
          self.layout().register_pcell(attr, value())
      except:
        pass

    # Register us the library with the technology name
    # If a library with that name already existed, it will be replaced then.
    self.register(library)

  
GSiP()