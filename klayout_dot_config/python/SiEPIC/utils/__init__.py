#################################################################################
#                SiEPIC Tools - utils                                           #
#################################################################################
'''
List of functions:


advance_iterator
get_library_names
get_technology_by_name
get_technology
load_Waveguides
load_Waveguides_by_Tech
load_Calibre
load_Monte_Carlo
load_DFT
load_FDTD_settings
load_GC_settings
get_layout_variables
enum
find_paths
selected_opt_in_text
select_paths
select_waveguides
select_instances
angle_b_vectors
inner_angle_b_vectors
angle_vector
angle_trunc
points_per_circle
arc
arc_wg
arc_wg_xy
arc_bezier
arc_to_waveguide
translate_from_normal
pt_intersects_segment
layout_pgtext
find_automated_measurement_labels
find_SEM_labels
find_siepictools_debug_text
etree_to_dict: XML parser
xml_to_dict
eng_str
svg_from_component
sample_function
pointlist_to_path


'''

from SiEPIC._globals import Python_Env
if Python_Env == "KLayout_GUI":
    from . import components

import pya

'''
from .. import _globals
if _globals.Python_Env == "KLayout_GUI":
    import pya
'''

# Python 2 vs 3 issues:  http://python3porting.com/differences.html
# Python 2: iterator.next()
# Python 3: next(iterator)
# Python 2 & 3: advance_iterator(iterator)
try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()


'''
Get Technology functions:
 - get_technology_by_name(tech_name)
 - get_technology()
 - get_layout_variables(), also returns layout, cell.

return:
TECHNOLOGY['dbu'] is the database unit
TECHNOLOGY['layer name'] is a LayerInfo object.

'''

'''
Read the layer table for a given technology.
Usage:
import SiEPIC.utils
SiEPIC.utils.get_technology_by_name('EBeam')
'''

#from functools import lru_cache
#@lru_cache(maxsize=None)
def get_library_names(tech_name, verbose=False):
    '''Returns a list of library names associated to the given technology name'''
    if verbose:
        print("get_library_names()")
    
    from .._globals import KLAYOUT_VERSION

    library_names = []
    if KLAYOUT_VERSION < 27:  #  technologies in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
        for lib_name in pya.Library.library_names():
            library = pya.Library.library_by_name(lib_name)
            if library:
                if tech_name == library.technology:
                    library_names.append(lib_name)
            else:
                print(' - library %s not working' % (lib_name) )

    if KLAYOUT_VERSION >= 27:  #  technologies in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
        libs = pya.Library.library_ids()
        for lib in libs:
            library = pya.Library.library_by_id(lib)
            if library:
                if tech_name in library.technologies():
                    library_names.append(library.name())
            else:
                print(' - library id %s not working' % (lib) )

    if verbose:
        print("get_library_names: tech=%s, lib: %s" % (tech_name, library_names))
    
    if not library_names:
        print("No libraries associated to {} technology".format(tech_name))
    
    return library_names

from functools import lru_cache
@lru_cache(maxsize=None)
def get_technology_by_name(tech_name, verbose=False):
    '''Get the SiEPIC-Tools technology associated to the given technology name'''
    if verbose:
        print("get_technology_by_name()")

    if not tech_name:
        pya.MessageBox.warning(
            "Problem with Technology", "Problem with active Technology: please activate a technology (not Default)", pya.MessageBox.Ok)
        return

    from .._globals import KLAYOUT_VERSION
    technology = {}
    technology['technology_name'] = tech_name
    if KLAYOUT_VERSION > 24:
        technology['dbu'] = pya.Technology.technology_by_name(tech_name).dbu
    else:
        technology['dbu'] = 0.001

    import os
    if KLAYOUT_VERSION > 24:
        lyp_file = pya.Technology.technology_by_name(tech_name).eff_layer_properties_file()
        technology['base_path'] = pya.Technology.technology_by_name(tech_name).base_path()
        if not technology['base_path']:
            raise Exception('Cannot find the technology "%s"' % tech_name)
    else:
        import fnmatch
        dir_path = pya.Application.instance().application_data_path()
        search_str = '*' + tech_name + '.lyp'
        matches = []
        for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
            for filename in fnmatch.filter(filenames, search_str):
                matches.append(os.path.join(root, filename))
        if matches:
            lyp_file = matches[0]
        else:
            raise Exception('Cannot find technology layer properties file %s' % search_str)

        # Load technology folder location
        technology['base_path'] = os.path.dirname(lyp_file)
        print('technology base path:%s' % technology['base_path'])

    # Find the Compact Model Library files    
    cml_files = []
    cml_paths = []
    cml_versions = []
    cml_names = []
    
    for file in os.listdir(technology['base_path']):
        if file.lower().endswith(".cml"):
            # Only store newest CMLs
            cml_name, cml_version = file.split('_v', 1)
            if not cml_name in cml_names:
                cml_files.insert(0, file)
                cml_paths.insert(0, os.path.join(technology['base_path'], file))
                cml_names.insert(0, cml_name)
                cml_versions.insert(0, 'v'+cml_version)
            elif cml_name in cml_names:
                cml_ind = cml_names.index(cml_name)
                if ('v'+cml_version) > cml_versions[cml_ind]:
                    cml_files[cml_ind] = file
                    cml_paths[cml_ind] = os.path.join(technology['base_path'], file)
                    cml_versions[cml_ind] = 'v'+cml_version
    
    if os.path.isdir(os.path.join(technology['base_path'], 'cml')):
        for file in os.listdir(os.path.join(technology['base_path'], 'cml')):
            if file.lower().endswith('.cml'):
                # Only store newest CMLs
                cml_name, cml_version = file.split('_v', 1)
                if not cml_name in cml_names:
                    cml_files.append(file)
                    cml_paths.append(os.path.join(technology['base_path'], 'cml', file))
                    cml_names.append(cml_name)
                    cml_versions.append('v'+cml_version)
                elif cml_name in cml_names:
                    cml_ind = cml_names.index(cml_name)
                    if ('v'+cml_version) > cml_versions[cml_ind]:
                        cml_files[cml_ind] = file
                        cml_paths[cml_ind] = os.path.join(technology['base_path'], 'cml', file)
                        cml_versions[cml_ind] =  'v'+cml_version
    
    if cml_files:
        technology['INTC_CML'] = cml_files[0]
        technology['INTC_CML_path'] = cml_paths[0]
        technology['INTC_CML_version'] = cml_files[0].replace(tech_name + '_', '')
        
        technology['INTC_CMLs'] = cml_files
        technology['INTC_CMLs_name'] = cml_names
        technology['INTC_CMLs_path'] = cml_paths
        technology['INTC_CMLs_version'] = ['v'+x.split('_v')[-1] for x in cml_files]
    else:
        technology['INTC_CML'] = ''
        technology['INTC_CML_path'] = ''
        technology['INTC_CML_version'] = ''
    
    # Layers:
    file = open(lyp_file, 'r')
    xml_dict = xml_to_dict(file.read())
    if("layer-properties-tabs" in xml_dict):
        #if multiple layer tabs are present then use layout.current_layer_list for index
        lv = pya.Application.instance().main_window().current_view()
        layer_dict = xml_dict["layer-properties-tabs"]['layer-properties'][lv.current_layer_list]['properties']
    else:
        layer_dict = xml_dict['layer-properties']['properties']
   
   
    file.close()

    def get_members(layer_dict, technology):
        if isinstance(layer_dict, list):
            for k in layer_dict:
                get_members(k, technology)
        elif 'group-members' in layer_dict:
            get_members(layer_dict['group-members'], technology)

        elif 'name' in layer_dict:
            layerInfo = layer_dict['source'].split('@')[0]
            if layer_dict['source'] != '*/*@*':
                technology[layer_dict['name']] = pya.LayerInfo(int(layerInfo.split('/')[0]), int(layerInfo.split('/')[1]))

        return technology

    get_members(layer_dict, technology)
        

    
    # Get library names
    technology['libraries'] = get_library_names(tech_name)

    
    return technology
# end of get_technology_by_name(tech_name)
# test example: give it a name of a technology, e.g., GSiP
# print(get_technology_by_name('EBeam'))
# print(get_technology_by_name('GSiP'))



def get_technology(verbose=False, query_activecellview_technology=False):
    '''Get the current Technology'''
    if verbose:
        print("get_technology()")
    from .._globals import KLAYOUT_VERSION
    technology = {}

    # defaults:
    technology['DevRec'] = pya.LayerInfo(68, 0)
    technology['Waveguide'] = pya.LayerInfo(1, 0)
    technology['Si'] = pya.LayerInfo(1, 0)
    technology['PinRec'] = pya.LayerInfo(69, 0)
    technology['Lumerical'] = pya.LayerInfo(733, 0)
    technology['Text'] = pya.LayerInfo(10, 0)
    technology_name = 'EBeam'

    if Python_Env == 'KLayout_GUI':
        lv = pya.Application.instance().main_window().current_view()
    else:
        lv = None

    if lv == None:
        # no layout open; return a default technology
        technology['dbu'] = 0.001
        technology['technology_name'] = technology_name
        
        # Get library names
        technology['libraries'] = get_library_names(technology_name)
        
        return technology

    # "lv.active_cellview().technology" crashes in KLayout 0.24.10 when loading a GDS file (technology not defined yet?) but works otherwise
    if KLAYOUT_VERSION > 24 or query_activecellview_technology or lv.title != '<empty>':
        technology_name = lv.active_cellview().technology
    
    return get_technology_by_name(technology_name)


def load_Waveguides():
    '''
    Load Waveguide configuration
    determine the technology from the layout
    These are technology specific, and located in the tech folder, named WAVEGUIDES.xml
    '''
    import os
    import fnmatch
    from . import get_technology
    TECHNOLOGY = get_technology()
    tech_name = TECHNOLOGY['technology_name']

    waveguides = load_Waveguides_by_Tech(tech_name, debug=False)

    return waveguides if waveguides else None

from functools import lru_cache
@lru_cache(maxsize=None)
def load_Waveguides_by_Tech(tech_name, debug=False):
    '''
    Load Waveguide configuration for specific technology
    These are technology specific, and located in the tech folder, named WAVEGUIDES.xml, and WAVEGUIDES_*.xml
    For KLayout <0.27, Look for this file for folders that contain 'tech_name'.lyt
    For KLayout 0.27+, Look in the technology folder, plus each library's folder.
    '''
    import os
    import fnmatch

    paths = []

    from .._globals import KLAYOUT_VERSION

    if KLAYOUT_VERSION >= 27:  #  technologies in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
        # Find the path for the technology
        # and find WAVEGUIDE.xml and WAVEGUIDE_*.xml files
        tech=pya.Technology.technology_by_name(tech_name)
        folder = tech.base_path()
        for root, dirnames, filenames in os.walk(folder, followlinks=True):
            if debug:
                print(' - %s, %s, %s, %s' % (root, dirnames, filenames, (fnmatch.filter(filenames, 'WAVEGUIDES.xml')+fnmatch.filter(filenames, 'WAVEGUIDES_*.xml'))))        
            [paths.append(os.path.join(root, filename))
             for filename in (fnmatch.filter(filenames, 'WAVEGUIDES.xml')) if fnmatch.filter(filenames, tech_name + '.lyt') ]
            [paths.append(os.path.join(root, filename))
             for filename in (fnmatch.filter(filenames, 'WAVEGUIDES_*.xml')) if fnmatch.filter(filenames, tech_name + '.lyt') ]

        # Find the paths for each Library that matches technology
        # and find WAVEGUIDE.xml and WAVEGUIDE_*.xml files
        libs = [pya.Library.library_by_id(lib) for lib in pya.Library.library_ids() if tech_name in pya.Library.library_by_id(lib).technologies()]
        libs = [lib for lib in libs if 'path' in dir(lib)]
        for lib in libs:
            for root, dirnames, filenames in os.walk(lib.path, followlinks=True):
                if debug:
                    print(' - %s, %s, %s, %s' % (root, dirnames, filenames, (fnmatch.filter(filenames, 'WAVEGUIDES.xml')+fnmatch.filter(filenames, 'WAVEGUIDES_*.xml'))))        
                [paths.append(os.path.join(root, filename))
                 for filename in (fnmatch.filter(filenames, 'WAVEGUIDES.xml'))  ] 
                [paths.append(os.path.join(root, filename))
                 for filename in (fnmatch.filter(filenames, 'WAVEGUIDES_*.xml'))  ] 
             

    if KLAYOUT_VERSION < 27:  #  technologies in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
        # Search all the sub-folders in KLayout
        # look for WAVEGUIDE xml files to find tech_name.lyt file
        # not very efficient.
        for root, dirnames, filenames in os.walk(pya.Application.instance().application_data_path(), followlinks=True):
            if debug:
                print(' - %s, %s, %s, %s' % (root, dirnames, filenames, (fnmatch.filter(filenames, 'WAVEGUIDES.xml')+fnmatch.filter(filenames, 'WAVEGUIDES_*.xml'))))        
                [paths.append(os.path.join(root, filename))
                 for filename in (fnmatch.filter(filenames, 'WAVEGUIDES.xml'))  ] 
                [paths.append(os.path.join(root, filename))
                 for filename in (fnmatch.filter(filenames, 'WAVEGUIDES_*.xml'))  ] 

    if debug:
        print(paths)

    # remove duplicates; keep unique paths
    paths = list(set(paths))

    if debug:
        print(paths)
        
    waveguides = []
    if paths:
        for path1 in paths:
            with open(path1, 'r') as file:
                waveguides1 = xml_to_dict(file.read())
                try:
                    if type(waveguides1['waveguides']['waveguide']) == list:
                        for waveguide in waveguides1['waveguides']['waveguide']:
                            waveguides.append(waveguide)
                    else:
                        waveguides.append(waveguides1['waveguides']['waveguide'])
                except:
                    pass
        for waveguide in waveguides:
            if 'component' in waveguide.keys():
                if not isinstance(waveguide['component'], list):
                    waveguide['component'] = [waveguide['component']]
            if not 'bezier' in waveguide.keys():
                waveguide['adiabatic'] = False
                waveguide['bezier'] = ''
            else:
                waveguide['adiabatic'] = True
            if not 'CML' in waveguide.keys():
                waveguide['CML'] = ''
            if not 'model' in waveguide.keys():
                waveguide['model'] = ''
    if not(waveguides):
        print('No waveguides found for technology=%s. Check that there exists a technology definition file %s.lyt and WAVEGUIDES.xml file' % (tech_name, tech_name) )
    
    if debug:
        print('- done: load_Waveguides_by_Tech.  Technology: %s' %(tech_name) )
    return waveguides if waveguides else None



def load_Calibre():
    '''
    Load Calibre configuration
    These are technology specific, and located in the tech folder, named CALIBRE.xml
    '''
    from . import get_technology
    TECHNOLOGY = get_technology()
    tech_name = TECHNOLOGY['technology_name']

    technology = {}
    technology['technology_name'] = tech_name
    technology['base_path'] = pya.Technology.technology_by_name(tech_name).base_path()

    import os
    import fnmatch
#    dir_path = pya.Application.instance().application_data_path()
    
    search_str = 'CALIBRE.xml'


    import fnmatch
    dir_path = technology['base_path']
    matches = []
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, search_str):
            matches.append(os.path.join(root, filename))
    if matches:
        CALIBRE_file = os.path.join(technology['base_path'], matches[-1])
        file = open(CALIBRE_file, 'r')
        print(CALIBRE_file)
        CALIBRE = xml_to_dict(file.read())
        print(CALIBRE)
        file.close()
        return CALIBRE
    else:
        return None



def load_Monte_Carlo():
    '''
    Load Monte Carlo configuration
    These are technology specific, and located in the tech folder, named MONTECARLO.xml
    '''
    import os
    import fnmatch
    from . import get_technology
    TECHNOLOGY = get_technology()
    tech_name = TECHNOLOGY['technology_name']
    paths = []
    for root, dirnames, filenames in os.walk(pya.Application.instance().application_data_path(), followlinks=True):
        [paths.append(os.path.join(root, filename))
         for filename in fnmatch.filter(filenames, 'MONTECARLO.xml') if tech_name in root]
    if paths:
        with open(paths[0], 'r') as file:
            montecarlo = xml_to_dict(file.read())
            montecarlo = montecarlo['technologies']['technology']
            if not isinstance(montecarlo, list):
                montecarlo = [montecarlo]
    return montecarlo if montecarlo else None



def load_Verification(debug=True):
    '''
    Load Verification rules
    These are technology specific, and located in the tech folder, named Verification.xml
    '''
    from .._globals import KLAYOUT_VERSION
    from . import get_technology
    TECHNOLOGY = get_technology()
    tech_name = TECHNOLOGY['technology_name']

    import os
    import fnmatch

    # then check for Verification.xml in the PDK Technology folder
    search_str = 'Verification.xml'
    if KLAYOUT_VERSION >= 27:  #  technologies in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
        # Find the path for the technology
        # and find DFT.xml file
        tech=pya.Technology.technology_by_name(tech_name)
        dir_path = tech.base_path()
    else:
        dir_path = pya.Application.instance().application_data_path()
    if debug:
        print(' - load_Verification, path: %s' %dir_path ) 
    matches = []
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, search_str):
            if tech_name in root:
                matches.append(os.path.join(root, filename))
    if matches:
        if debug:
            print(' - load_Verification, matches: %s' %matches ) 
        Verification_file = matches[0]
        file = open(Verification_file, 'r')
        Verification = xml_to_dict(file.read())
        file.close()
        return Verification
    else:
        return None




def load_DFT(debug=True):
    '''
    Load Design-for-Test (DFT) rules
    These are technology specific, and located in the tech folder, named DFT.xml
    '''
    from .._globals import KLAYOUT_VERSION
    from . import get_technology
    TECHNOLOGY = get_technology()
    tech_name = TECHNOLOGY['technology_name']

    import os
    import fnmatch

    # first check for filename_DFT.xml file in local directory
    mw = pya.Application.instance().main_window()
    layout_filename = mw.current_view().active_cellview().filename()
    filename = os.path.splitext(os.path.basename(layout_filename))[0]
    local_DFT_path = os.path.join(os.path.dirname(os.path.realpath(layout_filename)), filename+'_DFT.xml')
    print(' - checking local DFT path: %s' %local_DFT_path ) 
    if os.path.exists(local_DFT_path):
        matches = [local_DFT_path]
    else:
    # then check for DFT.xml in the PDK Technology folder
        search_str = 'DFT.xml'
        if KLAYOUT_VERSION >= 27:  #  technologies in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
            # Find the path for the technology
            # and find DFT.xml file
            tech=pya.Technology.technology_by_name(tech_name)
            dir_path = tech.base_path()
        else:
            dir_path = pya.Application.instance().application_data_path()
        if debug:
            print(' - load_DFT, path: %s' %dir_path ) 
        matches = []
        for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
            for filename in fnmatch.filter(filenames, search_str):
                if tech_name in root:
                    matches.append(os.path.join(root, filename))
    if matches:
        if debug:
            print(' - load_DFT, matches: %s' %matches ) 
        DFT_file = matches[0]
        file = open(DFT_file, 'r')
        DFT = xml_to_dict(file.read())
        file.close()
        return DFT
    else:
        return None



def load_FDTD_settings():
    '''
    Load FDTD settings
    These are technology specific, and located in the tech folder, named FDTD.xml
    '''
    from . import get_technology
    TECHNOLOGY = get_technology()
    tech_name = TECHNOLOGY['technology_name']

    import os
    import fnmatch
    dir_path = pya.Application.instance().application_data_path()
    search_str = 'FDTD.xml'
    matches = []
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, search_str):
            if tech_name in root:
                matches.append(os.path.join(root, filename))
    if matches:
        f = matches[0]
        file = open(f, 'r')
        FDTD = xml_to_dict(file.read())
        file.close()

        FDTD = FDTD['FDTD']
        FDTD1 = {}
        for k in FDTD['floats'].keys():
            FDTD1[k] = float(FDTD['floats'][k])
        for k in FDTD['strings'].keys():
            FDTD1[k] = FDTD['strings'][k]
        return FDTD1
    else:
        return None




def load_GC_settings():
    '''
    Load GC settings
    These are technology specific, and located in the tech folder, named GC.xml
    '''
    from . import get_technology
    TECHNOLOGY = get_technology()
    tech_name = TECHNOLOGY['technology_name']

    import os
    import fnmatch
    dir_path = pya.Application.instance().application_data_path()
    search_str = 'GC.xml'
    matches = []
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, search_str):
            if tech_name in root:
                matches.append(os.path.join(root, filename))
    if matches:
        f = matches[0]
        file = open(f, 'r')
        GC = xml_to_dict(file.read())
        file.close()

        GC = GC['GC']
        GC1 = {}
        for k in GC['floats'].keys():
            GC1[k] = float(GC['floats'][k])
        for k in GC['strings'].keys():
            GC1[k] = GC['strings'][k]
            #print(GC)
        return GC1
    else:
        return None


def get_layout_variables():
    '''For KLayout Application use only; gets TECHNOLOGY, Layout View, Layout, and current Cell'''
    from . import get_technology
    TECHNOLOGY = get_technology()

    # Configure variables to find in the presently selected cell:
    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        print("No view selected")
        raise UserWarning("No view selected. Make sure you have an open layout.")
    # Find the currently selected layout.
    ly = pya.Application.instance().main_window().current_view().active_cellview().layout()
    if ly == None:
        raise UserWarning("No layout. Make sure you have an open layout.")
    # find the currently selected cell:
    cv = pya.Application.instance().main_window().current_view().active_cellview()
    cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    if cell == None:
        raise UserWarning("No cell. Make sure you have an open layout.")

    ly.TECHNOLOGY = TECHNOLOGY
    return TECHNOLOGY, lv, ly, cell




def find_paths(layer, cell=None):
    '''Find all paths, full hierarachy scan, return polygons on top cell, for Verfication'''

    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        raise Exception("No view selected")
    if cell is None:
        ly = lv.active_cellview().layout()
        if ly == None:
            raise Exception("No active layout")
        cell = lv.active_cellview().cell
        if cell == None:
            raise Exception("No active cell")
    else:
        ly = cell.layout()

    selection = []
    itr = cell.begin_shapes_rec(ly.layer(layer))
    while not(itr.at_end()):
        if itr.shape().is_path():
            selection.append(itr.shape().path.transformed(itr.trans()))
        itr.next()

    return selection



def selected_opt_in_text():
    '''KLayout Application use. Return all selected opt_in Text labels.
    # example usage: selected_opt_in_text()[0].shape.text.string'''
    from . import get_layout_variables
    TECHNOLOGY, lv, ly, cell = get_layout_variables()

    selection = lv.object_selection
    selection = [o for o in selection if (not o.is_cell_inst())
                 and o.shape.is_text() and 'opt_in' in o.shape.text.string]
    return selection



def select_paths(layer, cell=None, verbose=None):
    '''# KLayout Application use. Return all selected paths. If nothing is selected, select paths automatically'''
    if verbose:
        print("SiEPIC.utils.select_paths: layer: %s" % layer)

    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        raise Exception("No view selected")

    if cell is None:
        ly = lv.active_cellview().layout()
        if ly == None:
            raise Exception("No active layout")
        cell = lv.active_cellview().cell
        if cell == None:
            raise Exception("No active cell")
    else:
        ly = cell.layout()

    selection = lv.object_selection
    if verbose:
        print("SiEPIC.utils.select_paths: selection, before: %s" % lv.object_selection)
    if selection == []:
        itr = cell.begin_shapes_rec(ly.layer(layer))
        itr_count = 0
        while not(itr.at_end()):
#            if verbose:
#                print("SiEPIC.utils.select_paths: itr: %s" % itr)
            itr_count += 1
            if itr.shape().is_path():
                if verbose:
                    print("SiEPIC.utils.select_paths: path: %s" % itr.shape())
                selection.append(pya.ObjectInstPath())
                selection[-1].layer = ly.layer(layer)
                selection[-1].shape = itr.shape()
                selection[-1].top = cell.cell_index()
                selection[-1].cv_index = 0
            itr.next()
        if verbose:
            print("SiEPIC.utils.select_paths: # shapes founded: %s" % itr_count)
        lv.object_selection = selection
    else:
        lv.object_selection = [o for o in selection if (
            not o.is_cell_inst()) and o.shape.is_path()]
    if verbose:
        print("SiEPIC.utils.select_paths: selection, after: %s" % lv.object_selection)
    return lv.object_selection



def select_waveguides(cell=None):
    '''KLayout Application use. 
    Return all selected waveguides. If nothing is selected, select waveguides automatically
    Returns all cell_inst'''

    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        raise Exception("No view selected")

    if cell is None:
        ly = lv.active_cellview().layout()
        if ly == None:
            raise Exception("No active layout")
        cell = lv.active_cellview().cell
        if cell == None:
            raise Exception("No active cell")
    else:
        ly = cell.layout()

    selection = lv.object_selection
    if selection == []:
        for instance in cell.each_inst():
            if instance.cell.basic_name() == "Waveguide":
                selection.append(pya.ObjectInstPath())
                selection[-1].top = cell.cell_index()
                selection[-1].append_path(pya.InstElement.new(instance))
        lv.object_selection = selection
    else:
        lv.object_selection = [o for o in selection if o.is_cell_inst(
        ) and o.inst().cell.basic_name() == "Waveguide"]

    return lv.object_selection



def select_instances(cell=None):
    '''# Return all selected instances.
    # Returns all cell_inst'''

    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        raise Exception("No view selected")
    if cell is None:
        ly = lv.active_cellview().layout()
        if ly == None:
            raise Exception("No active layout")
        cell = lv.active_cellview().cell
        if cell == None:
            raise Exception("No active cell")
    else:
        ly = cell.layout()

    selection = lv.object_selection
    if selection == []:
        for instance in cell.each_inst():
            selection.append(pya.ObjectInstPath())
            selection[-1].top = cell.cell_index()
            selection[-1].append_path(pya.InstElement.new(instance))
        lv.object_selection = selection
    else:
        lv.object_selection = [o for o in selection if o.is_cell_inst()]

    return lv.object_selection


def angle_b_vectors(u, v):
    '''Find the angle between two vectors (not necessarily the smaller angle)'''
    from math import atan2, pi
    return (atan2(v.y, v.x) - atan2(u.y, u.x)) / pi * 180


def inner_angle_b_vectors(u, v):
    '''Find the angle between two vectors (will always be the smaller angle)'''
    from math import acos, pi
    if (u.abs() * v.abs()) > 0:
        return acos((u.x * v.x + u.y * v.y) / (u.abs() * v.abs())) / pi * 180
    else:
        return 0
        

def angle_vector(u):
    '''Find the angle of a vector'''
    from math import atan2, pi
    return (atan2(u.y, u.x)) / pi * 180




def angle_trunc(a, trunc):
    '''Truncate the angle'''
    return ((a % trunc) + trunc) % trunc



from functools import lru_cache
@lru_cache(maxsize=None)
def points_per_circle(radius, dbu=None):
    '''Calculate the recommended number of points in a circle, based on
    http://stackoverflow.com/questions/11774038/how-to-render-a-circle-with-as-few-vertices-as-possible'''
    # radius in microns
    from math import acos, pi, ceil
    if dbu == None:
        from . import get_technology
        TECHNOLOGY = get_technology()
        err = TECHNOLOGY['dbu'] / 2  # in nm  (there was an error here for a few years: a 1000X factor)
    else:
        err = dbu / 2  # in nm 
        
    return int(ceil(pi / acos(1 - err / radius))) if radius > 1 else 10 # Lukas' derivation (same answer as below)
#    return int(ceil(2 * pi / acos(2 * (1 - err / radius)**2 - 1)))
#    return int(ceil(2 * pi / acos(2 * (1 - err / radius)**2 - 1))) if radius > 100 else 100


from functools import lru_cache
@lru_cache(maxsize=None)
def arc(r, theta_start, theta_stop):
    '''function to draw an arc of waveguide
    # radius: radius
    # w: waveguide width
    # length units in dbu
    # theta_start, theta_stop: angles for the arc
    # angles in degrees'''

    from math import pi, cos, sin
    from . import points_per_circle

    circle_fraction = abs(theta_stop - theta_start) / 360.0
    npoints = int(points_per_circle(r/1000) * circle_fraction)
    if npoints == 0:
        npoints = 1
    da = 2 * pi / npoints * circle_fraction  # increment, in radians
    pts = []
    th = theta_start / 360.0 * 2 * pi
    for i in range(0, npoints + 1):
        pts.append(pya.Point.from_dpoint(pya.DPoint(
            (r * cos(i * da + th)) / 1, (r * sin(i * da + th)) / 1)))
    return pts

from functools import lru_cache
@lru_cache(maxsize=None)
def arc_xy(x, y, r, theta_start, theta_stop, DevRec=None):
    '''function to draw an arc of waveguide
    # radius: radius
    # w: waveguide width
    # length units in dbu
    # theta_start, theta_stop: angles for the arc
    # angles in degrees'''

    from math import pi, cos, sin
    from . import points_per_circle

    circle_fraction = abs(theta_stop - theta_start) / 360.0
    npoints = int(points_per_circle(r/1000) * circle_fraction)
    if DevRec:
        npoints = int(npoints / 3)
    if npoints == 0:
        npoints = 1
    da = 2 * pi / npoints * circle_fraction  # increment, in radians
    pts = []
    th = theta_start / 360.0 * 2 * pi
    for i in range(0, npoints + 1):
        pts.append(pya.Point.from_dpoint(pya.DPoint(
            (x + r * cos(i * da + th)) / 1, (y + r * sin(i * da + th)) / 1)))
    return pts


from functools import lru_cache
@lru_cache(maxsize=None)
def arc_wg(radius, w, theta_start, theta_stop, DevRec=None):
    '''function to draw an arc of waveguide
    # radius: radius
    # w: waveguide width
    # length units in dbu
    # theta_start, theta_stop: angles for the arc
    # angles in degrees'''

    from math import pi, cos, sin
    from . import points_per_circle

    # print("SiEPIC.utils arc_wg")
    circle_fraction = abs(theta_stop - theta_start) / 360.0
    npoints = int(points_per_circle(radius/1000) * circle_fraction)
    if DevRec:
        npoints = int(npoints / 3)
    if npoints == 0:
        npoints = 1
    da = 2 * pi / npoints * circle_fraction  # increment, in radians
    pts = []
    th = theta_start / 360.0 * 2 * pi
    for i in range(0, npoints + 1):
        pts.append(pya.Point.from_dpoint(pya.DPoint(
            ((radius + w / 2) * cos(i * da + th)) / 1, ((radius + w / 2) * sin(i * da + th)) / 1)))
    for i in range(npoints, -1, -1):
        pts.append(pya.Point.from_dpoint(pya.DPoint(
            ((radius - w / 2) * cos(i * da + th)) / 1, ((radius - w / 2) * sin(i * da + th)) / 1)))
    return pya.Polygon(pts)


from functools import lru_cache
@lru_cache(maxsize=None)
def arc_wg_xy(x, y, r, w, theta_start, theta_stop, DevRec=None):
    '''function to draw an arc of waveguide
    # x, y: location of the origin
    # r: radius
    # w: waveguide width
    # length units in dbu
    # theta_start, theta_stop: angles for the arc
    # angles in degrees'''

    from math import pi, cos, sin
    from . import points_per_circle

    circle_fraction = abs(theta_stop - theta_start) / 360.0
    npoints = int(points_per_circle(r/1000) * circle_fraction)
    if DevRec:
        npoints = int(npoints / 3)
    if npoints == 0:
        npoints = 1
    da = 2 * pi / npoints * circle_fraction  # increment, in radians
    pts = []
    th = theta_start / 360.0 * 2 * pi
    for i in range(0, npoints + 1):
        pts.append(pya.Point.from_dpoint(pya.DPoint(
            (x + (r + w / 2) * cos(i * da + th)) / 1, (y + (r + w / 2) * sin(i * da + th)) / 1)))
    for i in range(npoints, -1, -1):
        pts.append(pya.Point.from_dpoint(pya.DPoint(
            (x + (r - w / 2) * cos(i * da + th)) / 1, (y + (r - w / 2) * sin(i * da + th)) / 1)))
    return pya.Polygon(pts)



from functools import lru_cache
@lru_cache(maxsize=None)
def arc_bezier(radius, start, stop, bezier, DevRec=None):
    '''Create a bezier curve. While there are parameters for start and stop in
    degrees, this is currently only implemented for 90 degree bends
    Radius in Database units (dbu)'''
    from math import sin, cos, pi
    from SiEPIC.utils import points_per_circle
    N = points_per_circle(radius/1000)/4
    bezier=float(bezier) # in case the input was a string
    if DevRec:
        N = int(N / 3)
    else:
        N = int(N)
    if N < 5:
      N = 100
    L = radius  # effective bend radius / Length of the bend
    diff = 1. / (N - 1)  # convert int to float
    xp = [0, (1 - bezier) * L, L, L]
    yp = [0, 0, bezier * L, L]
    xA = xp[3] - 3 * xp[2] + 3 * xp[1] - xp[0]
    xB = 3 * xp[2] - 6 * xp[1] + 3 * xp[0]
    xC = 3 * xp[1] - 3 * xp[0]
    xD = xp[0]
    yA = yp[3] - 3 * yp[2] + 3 * yp[1] - yp[0]
    yB = 3 * yp[2] - 6 * yp[1] + 3 * yp[0]
    yC = 3 * yp[1] - 3 * yp[0]
    yD = yp[0]

    pts = [pya.Point(-L, 0) + pya.Point(xD, yD)]
    for i in range(1, N - 1):
        t = i * diff
        pts.append(pya.Point(-L, 0) + pya.Point(t**3 * xA + t**2 * xB +
                                                t * xC + xD, t**3 * yA + t**2 * yB + t * yC + yD))
    pts.extend([pya.Point(0, L - 1), pya.Point(0, L)])
    return pts


def arc_to_waveguide(pts, width):
    '''Take a list of points and create a polygon of width 'width' '''
    return pya.Polygon(translate_from_normal(pts, -width / 2.) + translate_from_normal(pts, width / 2.)[::-1])


def translate_from_normal(pts, trans):
    '''Translate each point by its normal a distance 'trans' '''
    #  pts = [pya.DPoint(pt) for pt in pts]
    pts = [pt.to_dtype(1) for pt in pts]
    if len(pts) < 2:
        return pts    
    from math import cos, sin, pi
    d = 1. / (len(pts) - 1)
    a = angle_vector(pts[1] - pts[0]) * pi / 180 + (pi / 2 if trans > 0 else -pi / 2)
    tpts = [pts[0] + pya.DPoint(abs(trans) * cos(a), abs(trans) * sin(a))]

    for i in range(1, len(pts) - 1):
        dpt = (pts[i + 1] - pts[i - 1]) * (2 / d)
        tpts.append(pts[i] + pya.DPoint(-dpt.y, dpt.x) * (trans / 1 / dpt.abs()))

    a = angle_vector(pts[-1] - pts[-2]) * pi / 180 + (pi / 2 if trans > 0 else -pi / 2)
    tpts.append(pts[-1] + pya.DPoint(abs(trans) * cos(a), abs(trans) * sin(a)))

    # Make ends manhattan
    if abs(tpts[0].x - pts[0].x) > abs(tpts[0].y - pts[0].y):
        tpts[0].y = pts[0].y
    else:
        tpts[0].x = pts[0].x
    if abs(tpts[-1].x - pts[-1].x) > abs(tpts[-1].y - pts[-1].y):
        tpts[-1].y = pts[-1].y
    else:
        tpts[-1].x = pts[-1].x
#  return [pya.Point(pt) for pt in tpts]
    return [pt.to_itype(1) for pt in tpts]



def pt_intersects_segment(a, b, c):
    """ Check if point c intersects the segment defined by pts a and b
    How can you determine a point is between two other points on a line segment?
    http://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment
    by Cyrille Ka.  Check if c is between a and b? """
    cross = abs((c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y))
    if round(cross, 5) != 0:
        return False

    dot = (c.x - a.x) * (b.x - a.x) + (c.y - a.y) * (b.y - a.y)
    if dot < 0:
        return False
    return False if dot > (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) else True



def layout_pgtext(cell, layer, x, y, text, mag, inv=False):
    '''Add bubble to a cell
    Example
    cell = pya.Application.instance().main_window().current_view().active_cellview().cell
    layout_pgtext(cell, LayerInfo(10, 0), 0, 0, "test", 1)'''
    pcell = cell.layout().create_cell("TEXT", "Basic", {"text": text,
                                                        "layer": layer,
                                                        "mag": mag,
                                                        "inverse": inv})
    dbu = cell.layout().dbu
    cell.insert(pya.CellInstArray(pcell.cell_index(), pya.Trans(pya.Trans.R0, x / dbu, y / dbu)))




def find_automated_measurement_labels(topcell=None, LayerTextN=None, GUI=False):
    """return all opt_in labels from a cell
    requires a layout with Text labels on the layer LayerTextN
    the format of the labels is
       opt_in_<polarization>_<wavelength>_<type>_<deviceID>_<params>
         or 
       opt_<polarization>_<wavelength>_<type>_<deviceID>_<params>
         or
       elec_<deviceID>_<params>
         or
       pwb_<recipeID>_<params>
    for electrical-optical measurements, the deviceID on the electrical contact
    needs to match that of the optical input
       
    returns:
        text_out: HTML text
        opt_in: a Dictionary

    example usage:
    topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
    LayerText = pya.LayerInfo(10, 0)
    LayerTextN = topcell.layout().layer(LayerText)
    find_automated_measurement_labels(topcell, LayerTextN)
    """
    
    import string
    if LayerTextN == None:
        from . import get_technology, find_paths
        TECHNOLOGY = get_technology()
        dbu = TECHNOLOGY['dbu']
        LayerTextN = TECHNOLOGY['Text']
    if not topcell:
        lv = pya.Application.instance().main_window().current_view()
        if lv == None:
            print("No view selected")
            raise UserWarning("No view selected. Make sure you have an open layout.")
        # Find the currently selected layout.
        ly = pya.Application.instance().main_window().current_view().active_cellview().layout()
        if ly == None:
            raise UserWarning("No layout. Make sure you have an open layout.")
        # find the currently selected cell:
        cv = pya.Application.instance().main_window().current_view().active_cellview()
        topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
        if topcell == None:
            raise UserWarning("No cell. Make sure you have an open layout.")

    text_out = '% X-coord, Y-coord, Polarization, wavelength, type, deviceID, params <br>'
    dbu = topcell.layout().dbu

    if not type(LayerTextN)==int:
        LayerTextN = topcell.layout().layer(LayerTextN)
    iter = topcell.begin_shapes_rec(LayerTextN)
    i = 0
    texts = []  # pya Text, for Verification
    opt_in = []  # dictionary containing everything extracted from the opt_in labels.
    device_ids = set()
    duplicate = False
    while not (iter.at_end()):
        if iter.shape().is_text():
            text = iter.shape().text
            if text.string.find("opt") > -1:
                i += 1
                text2 = iter.shape().text.transformed(iter.itrans())
                texts.append(text2)
                if 'opt_in' in text.string:
                    # allow for either opt_ or opt_in_ formats
                    textlabel = text.string
                else:
                    textlabel = text.string.replace('opt_','opt_in_')
                fields = textlabel.split("_")
                while len(fields) < 7:
                    fields.append('comment')
                if GUI == True:
                    if fields[5] in device_ids and not duplicate:
                        error = pya.QDialog(pya.Application.instance().main_window())
    
                        #        wdg.setAttribute(pya.Qt.WA_DeleteOnClose)
                        error.setAttribute = pya.Qt.WA_DeleteOnClose
    
                        error.resize(200, 100)
                        error.move(1, 1)
                        grid = pya.QGridLayout(error)
                        windowlabel = pya.QLabel(error)
                        windowlabel.setText("Duplicate device-ids detected. Please make sure all device-ids are unique")
                        grid.addWidget(windowlabel, 2, 2, 4, 4)
                        error.show()
                        duplicate = True
                    else:
                        device_ids.add(fields[5])
                opt_in.append({'opt_in': textlabel, 'x': int(text2.x * dbu), 'y': int(text2.y * dbu), 'pol': fields[2], 
                        'wavelength': fields[3], 'type': fields[4], 
                        'deviceID': fields[5], 'params': fields[6:], 'Text': text2})
                params_txt = ''
                for f in fields[6:]:
                    params_txt += ', ' + str(f)
                text_out += "%s, %s, %s, %s, %s, %s%s<br>" % (int(text2.x * dbu), int(text2.y * dbu), fields[2], 
                        fields[3], fields[4], fields[5], params_txt)
        iter.next()

    text_out += "<br>"
    text_out += '% X-coord, Y-coord, deviceID, padName, params <br>'
    dbu = topcell.layout().dbu

    iter = topcell.begin_shapes_rec(LayerTextN)
    i = 0
    while not (iter.at_end()):
        if iter.shape().is_text():
            text = iter.shape().text
            if text.string.find("elec") > -1:
                i += 1
                text2 = iter.shape().text.transformed(iter.itrans())
                texts.append(text2)
                fields = text.string.split("_")
                while len(fields) < 4:
                    fields.append('comment')
                opt_in.append({'elec': text.string, 'x': int(text2.x * dbu), 'y': int(text2.y * dbu),
                               'deviceID': fields[1], 'params': fields[2:], 'Text': text2})
                params_txt = ''
                for f in fields[3:]:
                    params_txt += ', ' + str(f)
                text_out += "%s, %s, %s, %s%s<br>" % (int(text2.x * dbu), int(text2.y * dbu), fields[
                    1], fields[2], params_txt)
        iter.next()


    text_out += "<br>"
    text_out += '% X-coord, Y-coord, recipeID, params <br>'
    dbu = topcell.layout().dbu

    iter = topcell.begin_shapes_rec(LayerTextN)
    i = 0
    while not (iter.at_end()):
        if iter.shape().is_text():
            text = iter.shape().text
            if text.string.find("pwb") > -1:
                i += 1
                text2 = iter.shape().text.transformed(iter.itrans())
                texts.append(text2)
                fields = text.string.split("_")
                while len(fields) < 4:
                    fields.append('comment')
                opt_in.append({'pwb': text.string, 'x': int(text2.x * dbu), 'y': int(text2.y * dbu),
                               'recipeID': fields[1], 'params': fields[2:], 'Text': text2})
                params_txt = ''
                for f in fields[3:]:
                    params_txt += ', ' + str(f)
                text_out += "%s, %s, %s, %s%s<br>" % (int(text2.x * dbu), int(text2.y * dbu), fields[
                    1], fields[2], params_txt)
        iter.next()
    #text_out += "<br> Number of automated measurement labels: %s.<br>" % i
    #text_out += "<br> Number of sub-cells: %s<br>" % topcell.child_cells()

    return text_out, opt_in


try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()

def find_SEM_labels(topcell=None, LayerSEMN=None):
    '''example usage:
    # topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
    # LayerSEM = pya.LayerInfo(200, 0)
    # LayerSEMN = topcell.layout().layer(LayerSEM)
    # find_SEM_labels(topcell, LayerSEMN)'''
    import string
    if not LayerSEMN:
        from . import get_technology, find_paths
        TECHNOLOGY = get_technology()
        dbu = TECHNOLOGY['dbu']
        LayerSEMN = TECHNOLOGY['SEM']
    if not topcell:
        lv = pya.Application.instance().main_window().current_view()
        if lv == None:
            print("No view selected")
            raise UserWarning("No view selected. Make sure you have an open layout.")
        # Find the currently selected layout.
        ly = pya.Application.instance().main_window().current_view().active_cellview().layout()
        if ly == None:
            raise UserWarning("No layout. Make sure you have an open layout.")
        # find the currently selected cell:
        cv = pya.Application.instance().main_window().current_view().active_cellview()
        topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
        if topcell == None:
            raise UserWarning("No cell. Make sure you have an open layout.")

    text_out = 'SEM image locations <br>'
    dbu = topcell.layout().dbu
    iter = topcell.begin_shapes_rec(topcell.layout().layer(LayerSEMN))
    i = 0
    texts = []  # pya Text, for Verification
    while not(iter.at_end()):
        if iter.shape().is_box():
            box = iter.shape().box
            i += 1
            box2 = iter.shape().box.transformed(iter.itrans())
            texts.append(box2)
            text_out += "%s, %s<br>" % (int(box2.left * dbu), int(box2.bottom * dbu) )
        iter.next()
    text_out += "<br> Number of SEM boxes: %s.<br>" % i
    
    return text_out



def find_siepictools_debug_text(topcell=None, LayerTextN=None):
    '''example usage:
    # topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
    # LayerText = pya.LayerInfo(10, 0)
    # LayerTextN = topcell.layout().layer(LayerText)
    # find_siepictools_debug_text(topcell, LayerTextN)'''
    import string
    if not LayerTextN:
        from . import get_technology, find_paths
        TECHNOLOGY = get_technology()
        dbu = TECHNOLOGY['dbu']
        LayerTextN = TECHNOLOGY['Text']
    if not topcell:
        lv = pya.Application.instance().main_window().current_view()
        if lv == None:
            print("No view selected")
            raise UserWarning("No view selected. Make sure you have an open layout.")
        # Find the currently selected layout.
        ly = pya.Application.instance().main_window().current_view().active_cellview().layout()
        if ly == None:
            raise UserWarning("No layout. Make sure you have an open layout.")
        # find the currently selected cell:
        cv = pya.Application.instance().main_window().current_view().active_cellview()
        topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
        if topcell == None:
            raise UserWarning("No cell. Make sure you have an open layout.")

    text_out = 'Extracting SiEPIC-Tools verification debug text from layout:\n\n'
    dbu = topcell.layout().dbu
    iter = topcell.begin_shapes_rec(topcell.layout().layer(LayerTextN))
    i = 0
    texts = []  # pya Text, for Verification
    while not(iter.at_end()):
        if iter.shape().is_text():
            text = iter.shape().text
            if text.string.find("SiEPIC-Tools verification") > -1:
                i += 1
                text2 = iter.shape().text.transformed(iter.itrans())
                texts.append(text2)
                text_out += "%s: in %s\n" % (text2.string, iter.shape().cell.name)
        iter.next()
    text_out += "Number of verification labels: %s.\n" % i
    
    return text_out




def etree_to_dict(t):
    '''XML to Dict parser, from:
    https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary-in-python/10077069'''
    from collections import defaultdict
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def xml_to_dict(t):
    from xml.etree import cElementTree as ET
    try:
        e = ET.XML(t)
    except:
        raise UserWarning("Error in the XML file.")
    return etree_to_dict(e)


def eng_str(x):
    '''x input in meters
    output in meters, engineering notation, rounded to 1 nm'''

    import math
    EngExp_notation = 1  # 1 = "1.0e-6", 0 = "1.0u"
    x = round(x * 1E9) / 1E9
    y = abs(x)
    if y == 0:
        return '0'
    else:
        exponent = int(math.floor(math.log10(y)))
        engr_exponent = exponent - exponent % 3
        if engr_exponent == -3:
            str_engr_exponent = "m"
            z = y / 10**engr_exponent
        elif engr_exponent == -6:
            str_engr_exponent = "u"
            z = y / 10**engr_exponent
        elif engr_exponent == -9:
            str_engr_exponent = "n"
            z = y / 10**engr_exponent
        else:
            str_engr_exponent = ""
            z = y / 10**engr_exponent
        sign = '-' if x < 0 else ''
        if EngExp_notation:
            return sign + str(round(z,11)) + 'E' + str(engr_exponent)
#      return sign+ '%3.3f' % z +str(str_engr_exponent)
        else:
            return sign + str(round(z,11)) + str(str_engr_exponent)



def svg_from_component(component, filename, verbose=False):
    '''Save an SVG file for the component, for INTC icons'''
    #  from utils import get_technology
    TECHNOLOGY = get_technology()

    # get polygons from component
    polygons = component.get_polygons(include_pins=False)

    x, y = component.DevRec_polygon.bbox().center().x, component.DevRec_polygon.bbox().center().y
    print('x,y: %s, %s' % (x,y))
    width, height = component.DevRec_polygon.bbox().width(), component.DevRec_polygon.bbox().height()
    scale = max(width, height) / 0.64

    # These values are trial and error guesses, but they don't always work well
    s1, s2 = (64, 64 * height / width) if width > height else (64 * width / height, 64)
    x1, y1 = 5, 3

    polygons_vertices = [[[round((vertex.x - x) * 100. / scale + s1 / x1, 2), round((y - vertex.y) * 100. / scale + s2 / y1, 2)]
                          for vertex in p.each_point()] for p in [p.to_simple_polygon() for p in polygons]]
    for p in polygons_vertices:
        print(p)

    
    try:  # not sure why the first time it gives an error (Windows 8.1 lukas VM), Mustafa: svgwrite is not a module available in KL windows python
        import svgwrite
        dwg = svgwrite.Drawing(filename, size=(str(s1) + '%', str(s2) + '%'), debug=False)
    except:
        pass
    try:
        from imp import reload
        reload(svgwrite)
        dwg = svgwrite.Drawing(filename, size=(str(s1) + '%', str(s2) + '%'), debug=False)
    except:
        print(" SiEPIC.utils.svg_from_component: could not generate svg")
        return

    if 'Waveguide_color' in TECHNOLOGY:
        c = bytearray.fromhex(hex(TECHNOLOGY['Waveguide_color'])[4:10])
    else:
        c = [150, 50, 50]
    color = svgwrite.rgb(c[0], c[1], c[2], 'RGB')
    for i in range(0, len(polygons_vertices)):
        if verbose:
            print('polygon: %s' % polygons_vertices[i])
        p = dwg.add(dwg.polyline(polygons_vertices[i], fill=color, debug=False))  # stroke=color

    dwg.save()


from .. import _globals
if _globals.Python_Env == "KLayout_GUI":
    from .._globals import MODULE_NUMPY
    if MODULE_NUMPY:
        from .sampling import sample_function


def pointlist_to_path(pointlist, dbu):
    '''convert [[230.175,169.18],[267.0,169.18],[267.0,252.0],[133.0,252.0],[133.0,221.82],[140.175,221.82]]
    to pya.Path'''
    
    points = []
    for p in points:
        points.append(pya.Point(p[0],p[1]))
    path = pya.Path(points)
    return path


    