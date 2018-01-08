from ctypes import *
try:
  from numpy import *
except:
  print('Missing numpy')
  
import platform
import inspect
import time
import os
from pprint import pprint

INTEROPLIB = {}  # dict for DLL path, one for each product
iapi = {}        # dict for CDLL handle, one for each product
handles = {}     # dict for API handles, one for each product

# Mac OSX:
if platform.system() == 'Darwin':
  INTEROPLIB['interconnect'] = "/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Matlab/libinterop-api.1.dylib"
  INTEROPLIB['fdtd'] = "/Applications/Lumerical/FDTD/FDTD.app/Contents/API/Matlab/libinterop-api.1.dylib"

# Windows
if platform.system() == 'Windows': 
    import sys
    path = "C:\\Program Files\\Lumerical\\INTERCONNECT\\api\\python"
    if os.path.exists(path):
      if not path in sys.path:
        sys.path.append(path) # windows
      CWD = os.path.dirname(os.path.abspath(__file__))
      os.chdir(path) 
      INTEROPLIB['interconnect'] = path + "\\interopapi.dll"
    path = "C:\\Program Files\\Lumerical\\FDTD\\api\\python"
    if os.path.exists(path):
      if not path in sys.path:
        sys.path.append(path) # windows
      CWD = os.path.dirname(os.path.abspath(__file__))
      os.chdir(path) 
      INTEROPLIB['fdtd'] = path + "\\interopapi.dll"

class Session(Structure):
    _fields_ = [("p", c_void_p)]

class LumString(Structure):
    _fields_ = [("len", c_ulonglong), ("str", POINTER(c_char))]

class LumMat(Structure):
    _fields_ = [("mode", c_uint), 
                ("dim", c_ulonglong), 
                ("dimlst", POINTER(c_ulonglong)), 
                ("data", POINTER(c_double))]


## For incomplete types where the type is not defined before it's used.
## An example is the LumStruct that contains a member of type Any but the type Any is still undefined
## Review https://docs.python.org/2/library/ctypes.html#incomplete-types for more information.
class LumNameValuePair(Structure):
    pass

class LumStruct(Structure):
    pass

class LumList(Structure):
    pass

class ValUnion(Union):
    pass

class Any(Structure):
    pass 

LumNameValuePair._fields_ = [("name", LumString), ("value", POINTER(Any))]
LumStruct._fields_ = [("size", c_ulonglong), ("elements", POINTER(POINTER(Any)))]
LumList._fields_ = [("size", c_ulonglong), ("elements", POINTER(POINTER(Any)))]
ValUnion._fields_ = [("doubleVal", c_double), 
                     ("strVal", LumString), 
                     ("matrixVal", LumMat),
                     ("structVal", LumStruct),
                     ("nameValuePairVal", LumNameValuePair),
                     ("listVal", LumList)]
Any._fields_ = [("type", c_int), ("val", ValUnion)]

def initLib(product):
    print("lumapi: loading %s" % INTEROPLIB[product])
    iapi[product] = CDLL(INTEROPLIB[product])
    
    iapi[product].appOpen.restype = Session
    iapi[product].appOpen.argtypes = [c_char_p, POINTER(c_ulonglong)]
    
    iapi[product].appClose.restype = None
    iapi[product].appClose.argtypes = [Session]
    
    iapi[product].appEvalScript.restype = int
    iapi[product].appEvalScript.argtypes = [Session, c_char_p]
    
    iapi[product].appGetVar.restype = int
    iapi[product].appGetVar.argtypes = [Session, c_char_p, POINTER(POINTER(Any))]
    
    iapi[product].appPutVar.restype = int
    iapi[product].appPutVar.argtypes = [Session, c_char_p, POINTER(Any)]
    
    iapi[product].allocateLumDouble.restype = POINTER(Any)
    iapi[product].allocateLumDouble.argtypes = [c_double]
    
    iapi[product].allocateLumString.restype = POINTER(Any)
    iapi[product].allocateLumString.argtypes = [c_ulonglong, c_char_p]
    
    iapi[product].allocateLumMatrix.restype = POINTER(Any)
    iapi[product].allocateLumMatrix.argtypes = [c_ulonglong, POINTER(c_ulonglong)]
    
    iapi[product].allocateComplexLumMatrix.restype = POINTER(Any)
    iapi[product].allocateComplexLumMatrix.argtypes = [c_ulonglong, POINTER(c_ulonglong)]

    iapi[product].allocateLumNameValuePair.restype = POINTER(Any)
    iapi[product].allocateLumNameValuePair.argtypes = [c_ulonglong, c_char_p, POINTER(Any)]

    iapi[product].allocateLumStruct.restype = POINTER(Any)
    iapi[product].allocateLumStruct.argtypes = [c_ulonglong, POINTER(POINTER(Any))]

    iapi[product].allocateLumList.restype = POINTER(Any)
    iapi[product].allocateLumList.argtypes = [c_ulonglong, POINTER(POINTER(Any))]
    
    iapi[product].freeAny.restype = None
    iapi[product].freeAny.argtypes = [POINTER(Any)]
    
    return iapi

#if os.path.exists(INTEROPLIB):
#  iapi = initLib()

if platform.system() == 'Windows': 
    os.chdir(CWD) # windows (current working path)

class LumApiError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

def open(product):
    url="";
    if product == "interconnect":
        url = b"icc://localhost?server=true"
    elif product == "fdtd":
        url = b"fdtd://localhost?server=true"
    elif product == "mode":
        url = b"mode://localhost?server=true"
    elif product == "device":
        url = b"device://localhost?server=true"

    if len(url)==0:
        raise LumApiError("Invalid product name")

    iapi[product] = initLib(product)
    
    KeyType = c_ulonglong * 2
    k = KeyType()
    k[0] = 0
    k[1] = 0
    h[product] = iapi[product].appOpen(url,k)
    if not iapi[product].appOpened(h):
        raise LumApiError("Failed to launch application")

    return h

def close(handle):
    product = list(handles.keys())[list(handles.values()).index(handle)]
    iapi[product].appClose(handle)


def evalScript_test(handle, code):
    ec = iapi.appEvalScript(handle, code.encode())
    return ec

def evalScript(handle, code):
    ec = iapi.appEvalScript(handle, code.encode())
    if ec < 0:
        raise LumApiError("Failed to evaluate code")

def getVar(handle, varname):
    value = POINTER(Any)();

    ec = iapi.appGetVar(handle, varname.encode(), byref(value))
    if ec < 0:
        raise LumApiError("Failed to get variable")

    r = 0.
    valType = value[0].type

    if valType < 0:
        raise LumApiError("Failed to get variable")

    if valType == 0:
        ls = value[0].val.strVal
        r = '';
        for i in range(ls.len):r += ls.str[i].decode()
    elif valType == 1:
        r = float(value[0].val.doubleVal)
    elif valType == 2:
        r = unpackMatrix(value[0].val.matrixVal)
    elif valType == 4:
        r = GetTranslator.getStructMembers(value[0])
    elif valType == 5:
        r = GetTranslator.getListMembers(value[0])

    iapi.freeAny(value)

    return r

def putString(handle, varname, value):       
    try:
        v = str(value).encode()
    except:
        raise LumApiError("Unsupported data type")

    a = iapi.allocateLumString(len(v), v)
    ec = iapi.appPutVar(handle, varname.encode(), a)
    iapi.freeAny(a)

    if ec < 0: raise LumApiError("Failed to put variable")

def putMatrix(handle, varname, value):     
    a = packMatrix(value)

    ec = iapi.appPutVar(handle, varname.encode(), a)
    iapi.freeAny(a)

    if ec < 0: raise LumApiError("Failed to put variable")

def putDouble(handle, varname, value):     
    try:
        v = float(value)
    except:
        raise LumApiError("Unsupported data type")

    a = iapi.allocateLumDouble(v)

    ec = iapi.appPutVar(handle, varname.encode(), a)
    iapi.freeAny(a)

    if ec < 0: raise LumApiError("Failed to put variable")

def putStruct(handle, varname, values):
    nvlist = 0
    try:
        nvlist = PutTranslator.putStructMembers(values)
    except LumApiError as e:
        raise
    except:
        raise LumApiError("Unknown exception")

    a = PutTranslator.translateStruct(nvlist)

    ec = iapi.appPutVar(handle, varname.encode(), a)
    iapi.freeAny(a)

    if ec < 0: raise LumApiError("Failed to put variable")

def putList(handle, varname, values):
    llist = 0
    try:
        llist = PutTranslator.putListMembers(values)
    except LumApiError as e:
        raise
    except:
        raise LumApiError("Unknown exception")

    a = PutTranslator.translateList(llist)

    ec = iapi.appPutVar(handle, varname.encode(), a)
    iapi.freeAny(a)

    if ec < 0: raise LumApiError("Failed to put variable")

#### Support classes and functions ####
def packMatrix(value):
    try:
        if 'numpy.ndarray' in str(type(value)):
            v = value
        else:
            v = array(value, order='F')
    except:
        raise LumApiError("Unsupported data type")

    dim = c_ulonglong(v.ndim)
    DimList = c_ulonglong * v.ndim
    dl = DimList()
    for i in range(v.ndim): dl[i] = v.shape[i]
    v = v.reshape([v.size], order='F')

    if v.dtype == complex:
        a = iapi.allocateComplexLumMatrix(dim, dl)
        for i in range(v.size):
            a[0].val.matrixVal.data[i] = v[i].real
            a[0].val.matrixVal.data[i+v.size] = v[i].imag
    else:
        a = iapi.allocateLumMatrix(dim, dl)
        for i in range(v.size): a[0].val.matrixVal.data[i] = v[i]

    return a

def unpackMatrix(value):
    lm = value
    l = 1;
    dl = [0] * lm.dim
    for i in range(lm.dim):
        l *= lm.dimlst[i]
        dl[i] = lm.dimlst[i]

    if lm.mode == 1:
        r = empty(l, dtype=float, order='F')
        for i in range(l): r[i] = lm.data[i]
        r = r.reshape(dl, order='F')
    else :
        r = empty(l, dtype=complex, order='F')
        for i in range(l): r[i] = complex(lm.data[i], lm.data[i+l])
        r = r.reshape(dl, order='F')
    return r

class PutTranslator:
    @staticmethod
    def translateStruct(value):
        return iapi.allocateLumStruct(len(value), value)

    @staticmethod
    def translateList(values):
        return iapi.allocateLumList(len(values), values)

    @staticmethod
    def translate(value):
        if type(value) is str:
            v = str(value).encode()
            return iapi.allocateLumString(len(v), v)
        elif type(value) is float:
            return iapi.allocateLumDouble(float(value))
        elif 'numpy.ndarray' in str(type(value)):
            return packMatrix(value)
        elif type(value) is dict:
            return PutTranslator.translateStruct(PutTranslator.putStructMembers(value))
        elif type(value) is list:
            return PutTranslator.translateList(PutTranslator.putListMembers(value))
        else:
            raise LumApiError("Unsupported data type")

    @staticmethod
    def putStructMembers(value):
        nvlist = (POINTER(Any) * len(value))()
        index = 0;
        for key in value:
            nvlist[index] = iapi.allocateLumNameValuePair(len(key), key.encode(),
                                        PutTranslator.translate(value[key]))
            index+=1
        return nvlist

    @staticmethod
    def putListMembers(value):
        llist = (POINTER(Any) * len(value))()
        index = 0
        for v in value:
            llist[index] = PutTranslator.translate(v)
            index+=1
        return llist 

class GetTranslator:
    @staticmethod
    def translateString(strVal):
        ls = strVal
        r = '';
        for i in range(ls.len):
            r += ls.str[i].decode()
        return r

    @staticmethod
    def recalculateSize(size, elements):
       ptr = Any.from_address(addressof(elements[0]))
       return (POINTER(Any)*size).from_address(addressof(elements[0]))

    @staticmethod
    def translate(d, element, nested = True):
        if element.type == 0:
            return GetTranslator.translateString(element.val.strVal)
        elif element.type == 1:
            return element.val.doubleVal
        elif element.type == 2:
            return unpackMatrix(element.val.matrixVal)
        elif element.type == 3:
            name = GetTranslator.translateString(element.val.nameValuePairVal.name)
            d[name] = GetTranslator.translate(d, element.val.nameValuePairVal.value[0])
            return d
        elif element.type == 4:
            return GetTranslator.getStructMembers(element)
        elif element.type == 5:
            return GetTranslator.getListMembers(element)
        else:
            raise LumApiError("Unsupported data type")

    @staticmethod
    def getStructMembers(value):
        elements = GetTranslator.recalculateSize(value.val.structVal.size, 
                             value.val.structVal.elements)
        d = {}
        for index in range(value.val.structVal.size):
            d = GetTranslator.translate(d, Any.from_address(addressof(elements[index][0])))
        return d

    @staticmethod
    def getListMembers(value):
        d = []
        elements = GetTranslator.recalculateSize(value.val.listVal.size, 
                             value.val.listVal.elements)
        for index in range(value.val.listVal.size):
            s = []
            e = GetTranslator.translate(s, Any.from_address(addressof(elements[index][0])))
            if len(s):
                d.append(s)
            else:
                d.append(e)
        return d
