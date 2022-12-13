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


if __name__=="__main__":
    # Example 7.1 An illustration of the TDMA in one dimension
    # |20 −5  0  0  0| |φ1| = |1100|
    # |-5 15 −5  0  0| |φ2| = | 100|
    # |0  −5 15 −5  0| |φ3| = | 100|
    # |0  0  −5 15 −5| |φ4| = | 100|
    # |0  0   0 −5 10| |φ5| = | 100|
    A = np.array([[20, -5, 0, 0, 0], [-5, 15, -5, 0, 0], [0, -5, 15, -5, 0], [0, 0, -5, 15, -5], [0, 0, 0, -5, 10]])
    
    a = np.array([20, 15, 15, 15, 10])
    b = np.array([-5, -5, -5, -5])
    c = np.array([-5, -5, -5, -5])
    
    d = np.array([1100, 100, 100, 100, 100])
    φ1 = TDMA(a, b, c, d)
    φ2 = TDMA(A, d)
    print(φ1)
    print(φ1 == φ2)
    
    # a = np.array([40, 30, 30, 20])
    # b = np.array([-10, -10, -10, 0])
    # c = np.array([  0, -10, -10, -10])
    # d = np.array([2500.,  500.,  500.,  500.])
    # φ = TDMA(a, b, c, d)
    # print(φ)
    
    
    # Example 7.2 A two dimensional line-by-line application of the TDMA
    aN = np.array([[   0,   0,     0], [ 10, 10, 10], [10, 10, 10], [ 10, 10, 10]])    
    aS = np.array([[  10,   10,   10], [ 10, 10, 10], [10, 10, 10], [  0,  0,  0]])
    aW = np.array([[   0,   10,   10], [  0, 10, 10], [ 0, 10, 10], [  0, 10, 10]])
    aE = np.array([[  10,   10,    0], [ 10, 10,  0], [10, 10,  0], [ 10, 10,  0]])
    aP = np.array([[  40,   50,   40], [ 30, 40, 30], [30, 40, 30], [ 20, 30, 20]])
    Su = np.array([[2500, 2000, 2000], [500,  0,  0], [500, 0,  0], [500,  0,  0]])
    
    result = LBL_TDMA(aP, aN, aS, aW, aE, Su)
    print(result)