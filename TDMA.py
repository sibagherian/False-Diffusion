import numpy as np


def TDMA(a, b, c=np.array([]), d=np.array([])):
    """ TDMA Solver based on Malaleskara

    Args:
        a (float): Array of Digonal Coefficient
        b (float): Array of Upper Digonal Coefficient
        c (float): Array of Lower Digonal Coefficient
        d (float): Array of Righthand Side of System 
        In Terms of Ax = b
        A (float): Array of known matrix
        b (float): Array of RHS

    Returns:
        float: Array of result
    """
    if c.size == 0 and d.size == 0:
        aa, bb, cc = GetCoeff(a)
        return TDMASolver(aa, bb, cc, b)
    else:
        return TDMASolver(a, b, c, d)


def TDMASolver(a, b, c, d):
    numEqn = len(a)
    if (len(b) < len(a)):
        b = -np.append(b, 0)
    else:
        b = -b
    if (len(c) < len(a)):
        c = -np.insert(c, 0, 0)
    else:
        c = -c
        
    Q = np.zeros(numEqn, dtype='float')
    P = np.zeros(numEqn, dtype='float')
    result = np.zeros(numEqn, dtype='float')
    for i in range(numEqn):
        denom = (a[i] - c[i]*P[i-1])
        P[i] = b[i]/ denom
        Q[i] = (d[i]+c[i]*Q[i-1])/denom
    result[-1] = Q[-1]
    for i in range(numEqn-2, -1, -1):
        result[i] = P[i] * result[i+1] + Q[i]
    return result


def GetCoeff(A):
    a = np.diagonal(A)
    c = np.diagonal(A, -1)
    b = np.diagonal(A, 1)
    return a, b, c


def LBL_TDMA(aP: np.ndarray, aN: np.ndarray, aS: np.ndarray, aW: np.ndarray, 
            aE: np.ndarray, Su: np.ndarray, tol=1e-10)->np.ndarray:
    T_Sol = np.zeros_like(aP, dtype='float')
    sweep = aP.shape[1]
    while True:
        T_old = T_Sol.copy()
        for i in range(sweep):
            a =  aP[:, i]
            b = -aS[:, i]
            c = -aN[:, i]
            if i == 0:
                d = Su[:, i] + aE[:, i] * T_Sol[:, i+1]
            elif i == sweep-1:
                d = Su[:, i] + aW[:, i] * T_Sol[:, i-1]
            else:
                d = Su[:, i] + aE[:, i] * T_Sol[:, i+1] + aW[:, i] * T_Sol[:, i-1]
            T_Sol[:,i] = TDMA(a, b, c, d)
        if np.linalg.norm(T_Sol - T_old) < tol:
            break
    T_Sol = np.around(T_Sol, 3)
    return T_Sol
