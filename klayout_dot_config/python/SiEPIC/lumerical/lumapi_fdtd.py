from ctypes import *
try:
  from numpy import *
except:
  print('Missing numpy')
import platform
import inspect
import os

INTEROPLIB = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/libinterop-api.so.1"

# Mac OSX:
if platform.system() == 'Darwin':
  INTEROPLIB = "/Applications/Lumerical/FDTD/FDTD.app/Contents/API/Matlab/libinterop-api.1.dylib"

# Windows
if platform.system() == 'Windows': 
    import sys
    path = "C:\\Program Files\\Lumerical\\FDTD\\api\\python"
    if os.path.exists(path):
      if not path in sys.path:
        sys.path.append(path) # windows
      CWD = os.path.dirname(os.path.abspath(__file__))
      os.chdir(path) 
      INTEROPLIB = path + "\\interopapi.dll"

class Session(Structure):
    _fields_ = [("p", c_void_p)]

class LumString(Structure):
    _fields_ = [("len", c_ulonglong), ("str", POINTER(c_char))]

class LumMat(Structure):
    _fields_ = [("mode", c_uint), ("dim", c_ulonglong), ("dimlst", POINTER(c_ulonglong)), ("data", POINTER(c_double))]
    
class ValUnion(Union):
    _fields_ = [("doubleVal", c_double), ("strVal", LumString), ("matrixVal", LumMat)]

class Any(Structure):
    _fields_ = [("type", c_int),("val", ValUnion)]

def initLib():
    print("lumapi: loading %s" % INTEROPLIB)
    iapi = CDLL(INTEROPLIB)
    
    iapi.appOpen.restype = Session
    iapi.appOpen.argtypes = [c_char_p, POINTER(c_ulonglong)]
    
    iapi.appClose.restype = None
    iapi.appClose.argtypes = [Session]
    
    iapi.appEvalScript.restype = int
    iapi.appEvalScript.argtypes = [Session, c_char_p]
    
    iapi.appGetVar.restype = int
    iapi.appGetVar.argtypes = [Session, c_char_p, POINTER(POINTER(Any))]
    
    iapi.appPutVar.restype = int
    iapi.appPutVar.argtypes = [Session, c_char_p, POINTER(Any)]
    
    iapi.allocateLumDouble.restype = POINTER(Any)
    iapi.allocateLumDouble.argtypes = [c_double]
    
    iapi.allocateLumString.restype = POINTER(Any)
    iapi.allocateLumString.argtypes = [c_ulonglong, c_char_p]
    
    iapi.allocateLumMatrix.restype = POINTER(Any)
    iapi.allocateLumMatrix.argtypes = [c_ulonglong, POINTER(c_ulonglong)]
    
    iapi.allocateComplexLumMatrix.restype = POINTER(Any)
    iapi.allocateComplexLumMatrix.argtypes = [c_ulonglong, POINTER(c_ulonglong)]
    
    iapi.freeAny.restype = None
    iapi.freeAny.argtypes = [POINTER(Any)]
    
    return iapi

if os.path.exists(INTEROPLIB):
  iapi = initLib()

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
        url = "icc://localhost?server=true"
    elif product == "fdtd":
        url = "fdtd://localhost?server=true"
    elif product == "mode":
        url = "mode://localhost?server=true"
    elif product == "device":
        url = "device://localhost?server=true"

    if len(url)==0:
        raise LumApiError("Invalid product name")

    KeyType = c_ulonglong * 2
    k = KeyType()
    k[0] = 0
    k[1] = 0
    h = iapi.appOpen(url,k)
    if not iapi.appOpened(h):
        raise LumApiError("Failed to launch application")

    return h

def close(handle):
    iapi.appClose(handle)
    
def evalScript(handle, code):
    ec = iapi.appEvalScript(handle, code)
    if ec < 0:
        raise LumApiError("Failed to evaluate code")
    
def getVar(handle, varname):
    value = POINTER(Any)();

    ec = iapi.appGetVar(handle, varname, byref(value))
    if ec < 0:
        raise LumApiError("Failed to get variable")

    r = 0.
    valType = value[0].type
    if valType == 1:
        r = float(value[0].val.doubleVal)
    elif valType == 0:
        ls = value[0].val.strVal
        r = '';
        for i in range(ls.len):r += ls.str[i]
    elif valType == 2:
        lm = value[0].val.matrixVal
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
    
    iapi.freeAny(value)        
    return r

def putString(handle, varname, value):       
    try:
        v = str(value)
    except:
        raise LumApiError("Unsupported data type")

    a = iapi.allocateLumString(len(v), v)
    ec = iapi.appPutVar(handle, varname, a)
    iapi.freeAny(a)
    
    if ec < 0: raise LumApiError("Failed to put variable")
    
def putMatrix(handle, varname, value):     
    try:
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

    ec = iapi.appPutVar(handle, varname, a)
    iapi.freeAny(a)
    
    if ec < 0: raise LumApiError("Failed to put variable")
    
def putDouble(handle, varname, value):     
    try:
        v = float(value)
    except:
        raise LumApiError("Unsupported data type")
    
    a = iapi.allocateLumDouble(v)

    ec = iapi.appPutVar(handle, varname, a)
    iapi.freeAny(a)
    
    if ec < 0: raise LumApiError("Failed to put variable")
