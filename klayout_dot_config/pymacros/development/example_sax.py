# https://github.com/flaport/sax

from SiEPIC.install import install
install('sax')


def coupler(coupling=0.5):
    kappa = coupling**0.5
    tau = (1-coupling)**0.5
    sdict = sax.reciprocal({
        ("in0", "out0"): tau,
        ("in0", "out1"): 1j*kappa,
        ("in1", "out0"): 1j*kappa,
        ("in1", "out1"): tau,
    })
    return sdict

coupler(coupling=0.3)

print(coupler(coupling=0.3))

def waveguide(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0):
    import numpy as np

    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * np.pi * neff * length / wl
    amplitude = np.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission =  amplitude * np.exp(1j * phase)
    sdict = sax.reciprocal({("in0", "out0"): transmission})
    return sdict

waveguide(length=100.0)

mzi, _ = sax.circuit(
    netlist={
        "instances": {
            "lft": coupler,
            "top": waveguide,
            "rgt": coupler,
        },
        "connections": {
            "lft,out0": "rgt,in0",
            "lft,out1": "top,in0",
            "top,out0": "rgt,in1",
        },
        "ports": {
            "in0": "lft,in0",
            "in1": "lft,in1",
            "out0": "rgt,out0",
            "out1": "rgt,out1",
        },
    }
)

type(mzi)


import numpy as np

wl = np.linspace(1.53, 1.57, 1000)
result = mzi(wl=wl, lft={'coupling': 0.3}, top={'length': 200.0}, rgt={'coupling': 0.8})


# Plot using Plotly:
import plotly.express as px
import pandas as pd # https://pandas.pydata.org/docs/user_guide/10min.html

# Two lines:
t1 = np.abs(result['in0', 'out0'])**2
t2 = np.abs(result['in0', 'out1'])**2

df = pd.DataFrame(np.stack((t1, t2)).transpose(), index=wl, columns=['Output 1','Output 2'])
fig = px.line(df, labels={'index':'Wavelength', 'value':'Transmission'}, markers=True)
fig.show()

