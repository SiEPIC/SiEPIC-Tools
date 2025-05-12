# Lumerical INTERCONNECT Python integration

from . import interconnect, fdtd, mode, contraDirectionalCoupler
# Don't import load_lumapi on start-up, in case user doesn't have Lumerical tools installed

#from . import interconnect, fdtd, load_lumapi
#from . import interconnect, fdtd, lumapi_fdtd, lumapi_intc


#print('Lumerical INTERCONNECT Python integration')


def find_lumapi(path='', verbose=False):
    '''
    Function to find Lumerical, and import the lumapi package
    '''
    if verbose:
        print(f'SiEPIC.lumerical.find_lumapi, path {path}')
    import os, platform, sys, inspect
    if not os.path.exists(path):
        if platform.system() == 'Darwin':
            path_app = '/Applications'
        elif platform.system() == 'Linux':
            path_app = '/opt'
        elif platform.system() == 'Windows': 
            path_app = 'C:\\Program Files'
        else:
            print('Not a supported OS')
            return
        # Application folder paths containing Lumerical
        p = [s for s in os.listdir(path_app) if "lumerical" in s.lower()]
        
        # check sub-folders for lumapi.py
        import fnmatch

        def extract_version(s):
            import re
            match = re.search(r'v(\d+)', s)
            return int(match.group(1)) if match else -1
        for dir_path in p:
            search_str = 'lumapi.py'
            matches = []
            for root, dirnames, filenames in os.walk(os.path.join(path_app,dir_path), followlinks=True):
                for filename in fnmatch.filter(filenames, search_str):
                    matches.append(root)
            if matches:
                if verbose:
                    print(matches)
                # path = matches[0] # keep the first one found
                path = max(matches, key=extract_version) # pick the one with the highest version number

    if os.path.exists(path):
        if verbose:
            print('Lumerical lumapi.py path: %s' % path)
        if not path in sys.path:
            sys.path.append(path)
    else:
        raise Exception (f'cannot find Lumerical, after searching {path_app}')
        
    try:
        import lumapi
    except:
        raise Exception ('cannot import Lumerical')

