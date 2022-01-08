import numpy as np


def generate_spline(xn, yn):
    # CREATING MATRICES A and b:
    # THESE ARE FORMED FROM EQUATIONS 17, 18 AND 19 IN PDF FROM MARKDOWN
    # THIS IS EQUIVALENT TO STEP 1 IN THE EXAMPLE IN PDF
    n = len(xn)

    A = np.zeros(shape=(n, n))
    A[0, 0], A[0, 1] = 2 * (xn[1] - xn[0]), (xn[1] - xn[0])  # EQ.18 FROM PDF
    A[-1, -2], A[-1, -1] = (xn[-1] - xn[-2]), 2 * (xn[-1] - xn[-2])  # EQ.19

    b = np.zeros(shape=n)
    b[0] = 3 * (yn[1] - yn[0])  # EQ.18
    b[-1] = 3 * (yn[-1] - yn[-2])  # EQ.19

    for j in range(1, n - 1):  # EQ.17:
        A[j][j] = 2 / (xn[j] - xn[j - 1]) + 2 / (xn[j + 1] - xn[j])
        A[j][j - 1] = 1 / (xn[j] - xn[j - 1])
        A[j][j + 1] = 1 / (xn[j + 1] - xn[j])

        b[j] = 3 * ((yn[j] - yn[j - 1]) / (xn[j] - xn[j - 1]) ** 2 + (yn[j + 1] - yn[j]) / (xn[j + 1] - xn[j]) ** 2)

    # SOLVING THE MATRIX EQUATION:
    # EQUIVALENT TO STEP 2 IN THE EXAMPLE
    A_inv = np.linalg.inv(A)  # A_inv is the inverse of the matrix A
    vn = np.matmul(A_inv, b)  # vn is a vector containing the nodal gradients

    # FINDING CONSTANTS FOR CUBIC POLYNOMIALS FROM EQUATIONS 7,8, 13 & 14
    # EQUIVALENT TO STEP 3 IN THE EXAMPLE
    an, bn, cn, dn = np.ndarray(shape=(n - 1, 1)), np.ndarray(shape=(n - 1, 1)), np.ndarray(shape=(n - 1, 1)), np.ndarray(shape=(n - 1, 1))

    for j in range(n - 1):
        an[j] = yn[j]  # EQ.7
        bn[j] = vn[j]  # EQ.8
        cn[j] = 3 * (yn[j + 1] - yn[j]) / (xn[j + 1] - xn[j]) ** 2 - (vn[j + 1] + 2 * vn[j]) / (xn[j + 1] - xn[j])   # EQ.13
        dn[j] = - 2 * (yn[j + 1] - yn[j]) / (xn[j + 1] - xn[j]) ** 3 + (vn[j + 1] + vn[j]) / (xn[j + 1] - xn[j]) ** 2  #EQ.14

    # GENERATING COORDINATES OF THE POINTS ON THE CUBIC SPLINE
    # x IS A LIST OF EQUALLY SPACED POINTS BETWEEN THE X COORDINATES OF THE FIRST AND LAST NODE
    # q IS THE OUTPUT OF THE CUBIC SPLINE POLYNOMIALS AT POINTS WITHIN x
    x = np.linspace(xn[0], xn[-1], 10000)
    q = np.ndarray(shape=(10000, 1))

    i = 0
    for j in range(n - 1):
        while x[i] <= xn[j + 1]:
            q[i] = an[j] + bn[j] * (x[i] - xn[j]) + cn[j] * (x[i] - xn[j]) ** 2 + dn[j] * (x[i] - xn[j]) ** 3  # EQ.6
            if x[i] == xn[n - 1]:
                break
            else:
                i += 1
        if x[i] == xn[n - 1]:
            break

    q = np.array(q).transpose().tolist()[0]
    return x, q
