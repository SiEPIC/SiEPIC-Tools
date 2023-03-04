""" 
    ChirpedContraDC utils 

    Author: Jonathan Cauchon
    jonathan.cauchon.2@ulaval.ca
"""

from modules import *

"""
    Basic
"""

def clc():
    print ("\n"*10)


"""
    Linear algebra
"""

def switchTop(P):
    if P.ndim == 3:
        FF = P[:,:2,:2]
        FG = P[:,:2,2:]
        GF = P[:,2:,:2]
        GG = P[:,2:,2:]
        GG_ = np.linalg.inv(GG)

        H = np.zeros(P.shape, dtype=complex)

        H[:,:2,:2] = FF - np.matmul(FG, np.matmul(GG_, GF))
        H[:,:2,2:] = np.matmul(FG,GG_)
        H[:,2:,:2] = np.matmul(-GG_,GF)
        H[:,2:,2:] = GG_

        return H

    elif P.ndim == 4:
        FF = P[:,:,:2,:2]
        FG = P[:,:,:2,2:]
        GF = P[:,:,2:,:2]
        GG = P[:,:,2:,2:]
        GG_ = np.linalg.inv(GG)

        H = np.zeros(P.shape, dtype=complex)

        H[:,:,:2,:2] = FF - np.matmul(FG, np.matmul(GG_, GF))
        H[:,:,:2,2:] = np.matmul(FG,GG_)
        H[:,:,2:,:2] = np.matmul(-GG_,GF)
        H[:,:,2:,2:] = GG_

        return H


""" Matrix exponential
source: https://github.com/geoopt/geoopt/blob/master/geoopt/linalg/_expm.py

"""


def pade13(A):
    """ Computes Pade decomposition """

    b = (64764752532480000., 
        32382376266240000., 
        7771770303897600.,
        1187353796428800., 
        129060195264000., 
        10559470521600.,
        670442572800., 
        33522128640., 
        1323241920., 
        40840800., 
        960960.,
        16380., 
        182., 
        1.)

    ident = np.eye(A.shape[-1], dtype=A.dtype)
    A2 = np.matmul(A, A)
    A4 = np.matmul(A2, A2)
    A6 = np.matmul(A4, A2)
    U = np.matmul(
        A,
        np.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6
        + b[5] * A4
        + b[3] * A2
        + b[1] * ident )

    V = ( np.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2)
        + b[6] * A6
        + b[4] * A4
        + b[2] * A2
        + b[0] * ident )

    return U, V


def matrix_2_power(x, p):
    for _ in range(int(p)):
        x = np.matmul(x,x)
    return x


def expm(A): 
    A_fro = np.linalg.norm(A)

    # Scaling step
    n_squarings = np.clip(np.ceil(np.log(A_fro / 5.371920351148152) /0.6931471805599453), a_min=0, a_max=None)
    # n_squarings = np.clip(np.ceil(np.log(A_fro / 5.371920351148152) /0.6931471805599453), a_min=0, a_max=None)
    scaling = 2.0 ** n_squarings

    A_scaled = A / scaling

    # Pade 13 approximation
    U, V = pade13(A_scaled)
    P = U + V
    Q = -U + V

    R = np.linalg.solve(P, Q)  # solve P = Q*R
    expmA = matrix_2_power(R, n_squarings)
    return expmA

