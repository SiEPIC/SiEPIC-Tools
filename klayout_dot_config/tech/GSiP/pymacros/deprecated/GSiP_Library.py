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
 
Lukas 2023/11
 - compatibility with PyPI usage of KLayout
 
Lukas 2024/10
 - moving all the library loading code into SiEPIC.scripts.load_klayout_library
 
"""

from SiEPIC.scripts import load_klayout_library

load_klayout_library('GSiP', 'GSiP', 'SiEPIC Generic SiP, v1.1', 'gds/building_blocks','pymacros/pcells_GSiP', verbose=False)
