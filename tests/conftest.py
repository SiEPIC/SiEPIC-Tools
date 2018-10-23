import os
import sys

from lygadgets import pya, isGUI, isGSI

if not isGSI():
    # add the package because it is not installed in system python
    siepic_package_path = os.path.join(os.path.dirname(__file__), '..', 'klayout_dot_config', 'python')
    sys.path.append(siepic_package_path)


### Spoof a whole bunch of stuff related to pya GUI ###

class NS_Catcher(type):
    ''' All this does is override the pya namespace with this class '''
    def __init__(cls, name, bases, dct):
        setattr(pya, name, cls)
        super().__init__(name, bases, dct)

    def __getattr__(cls, attr):
        return PhonyClass()


class PhonyClass(metaclass=NS_Catcher):
    ''' It only ever gives instances of PhonyClass when called or as attributes.
        It is good for stifling those long chained calls like::

            pya.QFormBuilder().load(ui_file, pya.Application.instance().main_window()).findChild('ok').clicked(self.ok)

        That call will do nothing of course, but it also won't error.
    '''
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return PhonyClass()

    def __getattr__(self, attr):
        return PhonyClass()

    def __setattr__(self, attr, value):
        pass

if not isGUI():
    class QMessageBox(PhonyClass): pass

    class QMessageBox_StandardButton(PhonyClass): pass

    class QFile(PhonyClass): pass

    class QIODevice(PhonyClass): pass

    class QFormBuilder(PhonyClass): pass


    class PhonyInstance(PhonyClass):
        ''' This has to return a string sometimes '''
        def application_data_path(self):
            return '~/.klayout'

        def version(self):
            return '0.25.3'

    class Application(PhonyClass):
        instance = PhonyInstance


    class PCellDeclarationHelper(PhonyClass):
        ''' WARNING: this is not a GUI feature. It should work,
            but klayout.db apparently has no PCellDeclarationHelper!
        '''
        pass
