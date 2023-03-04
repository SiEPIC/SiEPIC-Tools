import sys, os, platform


# Start Lumerical INTERCONNECT

# Try using SiEPIC.lumerical
try:
    import SiEPIC
    isSiEPIC=True
except:
    isSiEPIC=False
    
if isSiEPIC:
    # Load Lumerical INTERCONNECT and Python API: 
    from SiEPIC import _globals
    from SiEPIC.lumerical.interconnect import run_INTC
    run_INTC()
    lumapi = _globals.LUMAPI
    if not lumapi:
        raise Exception ('SiEPIC.lumerical.interconnect.INTC_loaddesignkit: Cannot load Lumerical INTERCONNECT and Python integration (lumapi).')
else:
    
    # Lumerical Python API path on system
    
    cwd = os.getcwd()
    
    if platform.system() == 'Windows':
        try:
            lumapi_path = r'C:\\Program Files\\Lumerical\\v212\\api\\python'
            os.chdir(lumapi_path)
            sys.path.append(lumapi_path)
            import lumapi
        except FileNotFoundError:
            lumapi_path = r'C:\\Program Files\\Lumerical\\v221\\api\\python'
            os.chdir(lumapi_path)
            sys.path.append(lumapi_path)
            import lumapi
            
    else:
        try:
            lumapi_path = '/Applications/Lumerical/v212/api/python/'
            os.chdir(lumapi_path)
            sys.path.append(lumapi_path)
            import lumapi
    
        except FileNotFoundError:
            lumapi_path = '/Applications/Lumerical v212.app/Contents/API/Python/' #Jflag
            os.chdir(lumapi_path)
            sys.path.append(lumapi_path)
            import lumapi
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(os.path.join(lumapi_path,'lumapi.py')):
        print('Found lumapi path at' + ': ' +lumapi_path)
        sys.path.append(lumapi_path)
    else:
        print('lumapi path does not exist, edit lumapi_path variable')
        
    os.chdir(cwd)
    

def generate_dat(pol = 'TE', terminate = True):
    mode = lumapi.open('mode')
    
    # feed polarization into model
    if pol == 'TE':
        lumapi.evalScript(mode,"mode_label = 'TE'; mode_ID = '1';")
    elif pol == 'TM':
        lumapi.evalScript(mode,"mode_label = 'TM'; mode_ID = '2';")
        
    # run write sparams script
    lumapi.evalScript(mode,"cd('%s');"%dir_path)
    lumapi.evalScript(mode,'write_sparams;')
    
    if terminate == True:
        lumapi.close(mode)
    
    run_INTC()
    return

def run_INTC():
    intc = lumapi.open('interconnect')
    
    svg_file = "contraDC.svg"
    sparam_file = "ContraDC_sparams.dat"
    command ='cd("%s");'%dir_path
    command += 'switchtodesign; new; deleteall; \n'
    command +='addelement("Optical N Port S-Parameter"); createcompound; select("COMPOUND_1");\n'
    command += 'component = "contraDC"; set("name",component); \n' 
    command += 'seticon(component,"%s");\n' %(svg_file)
    command += 'select(component+"::SPAR_1"); set("load from file", true);\n'
    command += 'set("s parameters filename", "%s");\n' % (sparam_file)
    command += 'set("load from file", false);\n'
    command += 'set("passivity", "enforce");\n'
    
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 1",'Left',0.25)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (1, 1)
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 2",'Left',0.75)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (2, 2)
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 3",'Right',0.25)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (3, 3)
    command += 'addport(component, "%s", "Bidirectional", "Optical Signal", "%s",%s);\n' %("Port 4",'Right',0.75)
    command += 'connect(component+"::RELAY_%s", "port", component+"::SPAR_1", "port %s");\n' % (4, 4)
    command += 'addtolibrary;\n'
    
    lumapi.evalScript(intc, command)
    
    return