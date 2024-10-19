
if 0: 
    import os
    print(os.path.split('pymacros/pcells_GSiP')[-1])

if 1:
    import os 
    import SiEPIC
    # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
    import sys
    p = os.path.abspath(os.path.join(SiEPIC.__path__[0], '../..', 'tech'))
    sys.path.insert(0,p)
    #print (p)
    #print(sys.path)
    import GSiP



if 0:
    import importlib.util
    import sys
    from pathlib import Path
    path = '/Users/lukasc/Documents/GitHub/SiEPIC-Tools/klayout_dot_config/tech/GSiP/pymacros/pcells_GSiP/__init__.py'
    module_name = 'pcells_GSiP'
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Add it to sys.modules
    spec.loader.exec_module(module)  # Execute the module code
    print(dir(module))
    module.Ring_Filter_DB


