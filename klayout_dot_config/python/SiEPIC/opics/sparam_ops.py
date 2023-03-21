""" Functions operating on s-parameter matrices
"""
from typing import Optional
import numpy as np
from numpy import ndarray


def connect_s(
    A: ndarray,
    port_idx_A: int,
    B: Optional[ndarray],
    port_idx_B: int,
    create_composite_matrix: bool = True,
) -> ndarray:
    """
    connect two n-port networks' s-matrices together.

    specifically, connect port `port_idx_A` on network `A` to port `port_idx_B` on network
    `B`. The resultant network has nports = (A.rank + B.rank-2). This
    function operates on, and returns s-matrices. The function
    :func:`connect` operates on :class:`Network` types.

    Parameters
    -----------
    A : :class:`numpy.ndarray`
            S-parameter matrix of `A`, shape is fxnxn
    port_idx_A : int
            port index on `A` (port indices start from 0)
    B : :class:`numpy.ndarray`
            S-parameter matrix of `B`, shape is fxnxn
    port_idx_B : int
            port index on `B`

    Returns
    -------
    C : :class:`numpy.ndarray`
        new S-parameter matrix


    Notes
    -------
    internally, this function creates a larger composite network
    and calls the  :func:`innerconnect_s` function. see that function for more
    details about the implementation

    See Also
    --------
        connect : operates on :class:`Network` types
        innerconnect_s : function which implements the connection
            connection algorithm


    """

    if create_composite_matrix:
        if port_idx_A > A.shape[-1] - 1 or port_idx_B > B.shape[-1] - 1:
            raise (ValueError("port indices are out of range"))

        nf = A.shape[0]  # num frequency points
        nA = A.shape[1]  # num ports on A
        nB = B.shape[1]  # num ports on B
        nC = nA + nB  # num ports on C

        # create composite matrix, appending each sub-matrix diagonally
        C = np.zeros((nf, nC, nC), dtype=np.complex128)
        C[:, :nA, :nA] = A.copy()
        C[:, nA:, nA:] = B.copy()

        # call innerconnect_s() on composit matrix C
        mat_result = innerconnect_s(C, port_idx_A, nA + port_idx_B)
        return mat_result
    else:
        # call innerconnect_s() on non-composit matrix A
        return innerconnect_s(A, port_idx_A, port_idx_B)


def innerconnect_s(A: ndarray, port_idx_A: int, port_idx_B: int) -> ndarray:
    """
    connect two ports of a single n-port network's s-matrix.

    Specifically, connect port `port_idx_A` to port `port_idx_B` on `A`. This results in
    a (n-2)-port network.  This     function operates on, and returns
    s-matrices. The function :func:`innerconnect` operates on
    :class:`Network` types.

    Parameters
    -----------
    A : :class:`numpy.ndarray`
        S-parameter matrix of `A`, shape is fxnxn
    port_idx_A : int
        port index on `A` (port indices start from 0)
    port_idx_B : int
        port index on `A`

    Returns
    -------
    C : :class:`numpy.ndarray`
            new S-parameter matrix

    Notes
    -----
    The algorithm used to calculate the resultant network is called a
    'sub-network growth',  can be found in [#]_. The original paper
    describing the  algorithm is given in [#]_.

    References
    ----------
    - Compton, R.C.; , "Perspectives in microwave circuit analysis,"
    Circuits and Systems, 1989.,
    Proceedings of the 32nd Midwest Symposium on , vol., no., pp.716-718 vol.2, 14-16
    Aug 1989.
    http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167

    - Filipsson, Gunnar;
    "A New General Computer Algorithm for S-Matrix Calculation of Interconnected
    Multiports," Microwave Conference, 1981. 11th European , vol., no., pp.700-704,
    7-11 Sept. 1981.
    http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4131699&isnumber=4131585
    """

    if port_idx_A > A.shape[-1] - 1 or port_idx_B > A.shape[-1] - 1:
        raise (ValueError("port indices are out of range"))

    nA = A.shape[1]  # num of ports on input s-matrix

    # create an empty s-matrix, to store the result
    C = np.zeros(shape=A.shape, dtype=np.complex128)

    # A[:, i,j] = (2000, 2,2)

    # loop through ports and calulates resultant s-parameters
    for i in range(nA):
        for j in range(nA):
            C[:, i, j] = (
                A[:, i, j]
                * (
                    A[:, port_idx_B, port_idx_B] * A[:, port_idx_A, port_idx_A]
                    - (A[:, port_idx_B, port_idx_A] - 1)
                    * (A[:, port_idx_A, port_idx_B] - 1)
                )
                + A[:, port_idx_A, j]
                * A[:, i, port_idx_B]
                * (A[:, port_idx_B, port_idx_A] - 1)
                - A[:, port_idx_A, j]
                * A[:, i, port_idx_A]
                * A[:, port_idx_B, port_idx_B]
                - A[:, i, port_idx_B]
                * A[:, port_idx_B, j]
                * A[:, port_idx_A, port_idx_A]
                + A[:, port_idx_B, j]
                * A[:, i, port_idx_A]
                * (A[:, port_idx_A, port_idx_B] - 1)
            ) / (
                A[:, port_idx_B, port_idx_B] * A[:, port_idx_A, port_idx_A]
                - (A[:, port_idx_B, port_idx_A] - 1)
                * (A[:, port_idx_A, port_idx_B] - 1)
            )

    # remove ports that were `connected`
    C = np.delete(C, (port_idx_A, port_idx_B), 1)
    C = np.delete(C, (port_idx_A, port_idx_B), 2)

    # ignore all from C[:,port_idx_A,:], and C[:,port_idx_B,:]
    # ignore all from C[:,:,port_idx_A], and C[:,:,port_idx_B]

    return C


def v_broadcast_sim(A: np.ndarray, port_idx_A: int, port_idx_B: int) -> np.ndarray:

    if port_idx_A > A.shape[-1] - 1 or port_idx_B > A.shape[-1] - 1:
        raise (ValueError("port indices are out of range"))

    nA = A.shape[1]  # num of ports on input s-matrix

    # create an empty s-matrix, to store the result
    C = np.zeros(shape=A.shape, dtype=np.complex128)

    # fundamental elements to broadcast
    _terms = {
        "a": A[:, port_idx_B, port_idx_B],
        "b": A[:, port_idx_A, port_idx_A],
        "c": A[:, port_idx_B, port_idx_A] - 1,
        "d": A[:, port_idx_A, port_idx_B] - 1,
        "e": np.full(
            (A.shape[0], nA, nA), np.reshape(A[:, port_idx_A, :nA], (A.shape[0], 1, nA))
        ),
        "f": np.full(
            (A.shape[0], nA, nA), np.reshape(A[:, :nA, port_idx_B], (A.shape[0], nA, 1))
        ),
        "g": np.full(
            (A.shape[0], nA, nA), np.reshape(A[:, :nA, port_idx_A], (A.shape[0], nA, 1))
        ),
        "h": np.full(
            (A.shape[0], nA, nA), np.reshape(A[:, port_idx_B, :nA], (A.shape[0], 1, nA))
        ),
    }

    _interm_terms = {
        "term1": np.full(
            (A.shape[0], nA, nA),
            np.reshape(
                (_terms["a"] * _terms["b"] - (_terms["c"] * _terms["d"])),
                (A.shape[0], 1, 1),
            ),
        ),
        "term2": _terms["e"]
        * _terms["f"]
        * np.full((A.shape[0], nA, nA), np.reshape(_terms["c"], (A.shape[0], 1, 1))),
        "term3": _terms["e"]
        * _terms["g"]
        * np.full((A.shape[0], nA, nA), np.reshape(_terms["a"], (A.shape[0], 1, 1))),
        "term4": _terms["f"]
        * _terms["h"]
        * np.full((A.shape[0], nA, nA), np.reshape(_terms["b"], (A.shape[0], 1, 1))),
        "term5": _terms["h"]
        * _terms["g"]
        * np.full((A.shape[0], nA, nA), np.reshape(_terms["d"], (A.shape[0], 1, 1))),
    }

    # A[:, i,j] = (2000, 2,2)

    # loop through ports and calulates resultant s-parameters
    C = (
        A * _interm_terms["term1"]
        + _interm_terms["term2"]
        - _interm_terms["term3"]
        - _interm_terms["term4"]
        + _interm_terms["term5"]
    ) / _interm_terms["term1"]

    # remove ports that were `connected`
    C = np.delete(C, (port_idx_A, port_idx_B), 1)
    C = np.delete(C, (port_idx_A, port_idx_B), 2)

    # ignore all from C[:,port_idx_A,:], and C[:,port_idx_B,:]
    # ignore all from C[:,:,port_idx_A], and C[:,:,port_idx_B]

    return C
