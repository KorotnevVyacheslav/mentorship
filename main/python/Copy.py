import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D

# o = m_epsilon_plus(1 , -100j + 1,  -100j + 1 , 1 , -100j + 1 , 1 )
# print(o)


def exp_phase(a, phi, epsilon, mu):
    """Changing phase to -2*i*a*kz
    a - Thickness
    phi - angle of incidence
    epsilon - dielectric coefficient
    mu - magnetic coefficient"""
    kz = np.sqrt(0j + epsilon * mu - np.sin(phi) ** 2)
    return np.exp(-1j * a * kz)


def m_epsilon_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Coefficient for intermediate calculations m+ for TM mode
    phi - angle of incidence
    e2 - phase change in SiO2
    epsilon_2 - dielectric coefficient of SiO2
    mu_2 - magnetic coefficient of Si
    epsilon_3 - dielectric coefficient of Si
    mu_3 - magnetic coefficient of Si
    """
    k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
    k3z = np.sqrt(0j + epsilon_3 * mu_3 - np.sin(phi) ** 2)
    if (k3z * epsilon_2 - k2z * epsilon_3) == 0:
        temp = 0
    else:
        temp = (k3z * epsilon_2 + k2z * epsilon_3) / (k3z * epsilon_2 - k2z * epsilon_3)
    return 1 + temp / (e2 ** 2)


def m_epsilon_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Coefficient for intermediate calculations m- for TM mode
    phi - angle of incidence
    e2 - phase change in SiO2
    epsilon_2 - dielectric coefficient of SiO2
    mu_2 - magnetic coefficient of Si
    epsilon_3 - dielectric coefficient of Si
    mu_3 - magnetic coefficient of Si
    """
    k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
    k3z = np.sqrt(0j + epsilon_3 * mu_3 - np.sin(phi) ** 2)
    if (k3z * epsilon_2 - k2z * epsilon_3) == 0:
        temp = 0
    else:
        temp = (k3z * epsilon_2 + k2z * epsilon_3) / (k3z * epsilon_2 - k2z * epsilon_3)
    return 1 - temp / (e2 ** 2)


def m_mu_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Coefficient for intermediate calculations m+ for TE mode
    phi - angle of incidence
    e2 - phase change in SiO2
    epsilon_2 - dielectric coefficient of SiO2
    mu_2 - magnetic coefficient of Si
    epsilon_3 - dielectric coefficient of Si
    mu_3 - magnetic coefficient of Si
    """
    k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
    k3z = np.sqrt(0j + epsilon_3 * mu_3 - np.sin(phi) ** 2)
    if (k3z * mu_2 - k2z * mu_3) == 0:
        temp = 0
    else:
        temp = (k3z * mu_2 + k2z * mu_3) / (k3z * mu_2 - k2z * mu_3)
    return 1 + temp / (e2 ** 2)


def m_mu_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Coefficient for intermediate calculations m- for TE mode
    phi - angle of incidence
    e2 - phase change in SiO2
    epsilon_2 - dielectric coefficient of SiO2
    mu_2 - magnetic coefficient of Si
    epsilon_3 - dielectric coefficient of Si
    mu_3 - magnetic coefficient of Si
    """
    k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
    k3z = np.sqrt(0j + epsilon_3 * mu_3 - np.sin(phi) ** 2)
    if (k3z * mu_2 - k2z * mu_3) == 0:
        temp = 0
    else:
        temp = (k3z * mu_2 + k2z * mu_3) / (k3z * mu_2 - k2z * mu_3)
    return 1 - temp / (e2 ** 2)


def r_TM(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Coefficient of reflection for TM mode
    phi - angle of incidence
    e1 - phase change in Si
    e2 - phase change in SiO2
    epsilon_1 = 1 - dielectric coefficient of Air
    mu_1 = 1 - magnetic coefficient of Air
    epsilon - dielectric coefficient of HRIM
    mu - magnetic coefficient of HRIM
    epsilon_2 - dielectric coefficient of SiO2
    mu_2 - magnetic coefficient of Si
    epsilon_3 - dielectric coefficient of Si
    mu_3 - magnetic coefficient of Si
    """

    def n_epsilon_plus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Coefficient for intermediate calculations n+ for TM mode
        phi - angle of incidence
        e1 - phase change in Si
        e2 - phase change in SiO2
        epsilon - dielectric coefficient of HRIM
        mu - magnetic coefficient of HRIM
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_epsilon_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_epsilon_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        # kz = np.sqrt(0j + epsilon * mu - np.sin(phi) ** 2)
        kz = np.sqrt(0j + epsilon * mu)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        if (k2z * epsilon * m_plus - kz * epsilon_2 * m_minus) == 0:
            temp = 0
        else:
            temp = (k2z * epsilon * m_plus + kz * epsilon_2 * m_minus) / (
                k2z * epsilon * m_plus - kz * epsilon_2 * m_minus
            )
        return 1 + temp / (e1 ** 2)

    def n_epsilon_minus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Coefficient for intermediate calculations n- for TM mode
        phi - angle of incidence
        e1 - phase change in Si
        e2 - phase change in SiO2
        epsilon - dielectric coefficient of HRIM
        mu - magnetic coefficient of HRIM
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_epsilon_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_epsilon_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        # kz = np.sqrt(0j + epsilon * mu - np.sin(phi) ** 2)
        kz = np.sqrt(0j + epsilon * mu)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        if (k2z * epsilon * m_plus - kz * epsilon_2 * m_minus) == 0:
            temp = 0
        else:
            temp = (k2z * epsilon * m_plus + kz * epsilon_2 * m_minus) / (
                k2z * epsilon * m_plus - kz * epsilon_2 * m_minus
            )
        return 1 - temp / (e1 ** 2)

    n_plus = n_epsilon_plus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3)
    n_minus = n_epsilon_minus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3)
    k1z = np.sqrt(0j + 1 - np.sin(phi) ** 2)
    # kz = np.sqrt(0j + epsilon * mu - np.sin(phi) ** 2)
    kz = np.sqrt(0j + epsilon * mu)
    r = (kz * n_plus - k1z * epsilon * n_minus) / (kz * n_plus + k1z * epsilon * n_minus)
    return r


def r_TE(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Coefficient of reflection for TE mode
    phi - angle of incidence
    e1 - phase change in Si
    e2 - phase change in SiO2
    epsilon_1 = 1 - dielectric coefficient of Air
    mu_1 = 1 - magnetic coefficient of Air
    epsilon - dielectric coefficient of HRIM
    mu - magnetic coefficient of HRIM
    epsilon_2 - dielectric coefficient of SiO2
    mu_2 - magnetic coefficient of Si
    epsilon_3 - dielectric coefficient of Si
    mu_3 - magnetic coefficient of Si
    """

    def n_mu_plus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Coefficient for intermediate calculations n+ for TE mode
        phi - angle of incidence
        e1 - phase change in Si
        e2 - phase change in SiO2
        epsilon - dielectric coefficient of HRIM
        mu - magnetic coefficient of HRIM
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_mu_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_mu_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        # kz = np.sqrt(0j + epsilon * mu - np.sin(phi) ** 2)
        kz = np.sqrt(0j + epsilon * mu)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        if (k2z * mu * m_plus - kz * mu_2 * m_minus) == 0:
            temp = 0
        else:
            temp = (k2z * mu * m_plus + kz * mu_2 * m_minus) / (
                k2z * mu * m_plus - kz * mu_2 * m_minus
            )
        return 1 + temp / (e1 ** 2)

    def n_mu_minus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Coefficient for intermediate calculations n- for TE mode
        phi - angle of incidence
        e1 - phase change in Si
        e2 - phase change in SiO2
        epsilon - dielectric coefficient of HRIM
        mu - magnetic coefficient of HRIM
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_mu_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_mu_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        # kz = np.sqrt(0j + epsilon * mu - np.sin(phi) ** 2)
        kz = np.sqrt(0j + epsilon * mu)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        if (k2z * mu * m_plus - kz * mu_2 * m_minus) == 0:
            temp = 0
        else:
            temp = (k2z * mu * m_plus + kz * mu_2 * m_minus) / (
                k2z * mu * m_plus - kz * mu_2 * m_minus
            )
        return 1 - temp / (e1 ** 2)

    n_plus = n_mu_plus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3)
    n_minus = n_mu_minus(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3)
    k1z = np.sqrt(0j + 1 - np.sin(phi) ** 2)
    # kz = np.sqrt(0j + epsilon * mu - np.sin(phi) ** 2)
    kz = np.sqrt(0j + epsilon * mu)
    r = (kz * n_plus - k1z * mu * n_minus) / (kz * n_plus + k1z * mu * n_minus)
    return r


def solutions(phi, a, b, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Finding epsilons and mus
    phi - angle of incidence
    a - thickness of HRIM
    b - thickness of SiO2
    epsilon_1 = 1 - dielectric coefficient of Air
    mu_1 = 1 - magnetic coefficient of Air
    epsilon_2 - dielectric coefficient of SiO2
    mu_2 - magnetic coefficient of Si
    epsilon_3 - dielectric coefficient of Si
    mu_3 - magnetic coefficient of Si
    """

    def a_epsilon(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Second coefficient in quadratic equation with respect to sqrt( epsilon ) for TM mode
        phi - angle of incidence
        a - thickness of HRIM
        e2 - phase change in SiO2
        epsilon_1 = 1 - dielectric coefficient of Air
        mu_1 = 1 - magnetic coefficient of Air
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_epsilon_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_epsilon_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        k1z = np.sqrt(0j + 1 - np.sin(phi) ** 2)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        r = (1 / k1z) + (epsilon_2 / k2z) * (m_minus / m_plus)
        return -r

    def b_epsilon(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Third coefficient in quadratic equation with respect to sqrt( epsilon ) for TM mode
        phi - angle of incidence
        a - thickness of HRIM
        e2 - phase change in SiO2
        epsilon_1 = 1 - dielectric coefficient of Air
        mu_1 = 1 - magnetic coefficient of Air
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_epsilon_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_epsilon_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        k1z = np.sqrt(0j + 1 - np.sin(phi) ** 2)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        return (1 / k1z) * (epsilon_2 / k2z) * (m_minus / m_plus)

    def c_epsilon(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Fourth coefficient in quadratic equation with respect to sqrt( epsilon ) for TM mode
        phi - angle of incidence
        a - thickness of HRIM
        e2 - phase change in SiO2
        epsilon_1 = 1 - dielectric coefficient of Air
        mu_1 = 1 - magnetic coefficient of Air
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_epsilon_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_epsilon_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        k1z = np.cos(phi)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        return ((1 / k1z) + (epsilon_2 / k2z) * (m_minus / m_plus)) / (1j * a)

    def a_mu(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Second coefficient in quadratic equation with respect to sqrt( epsilon ) for TE mode
        phi - angle of incidence
        a - thickness of HRIM
        e2 - phase change in SiO2
        epsilon_1 = 1 - dielectric coefficient of Air
        mu_1 = 1 - magnetic coefficient of Air
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_mu_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_mu_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        k1z = np.sqrt(0j + 1 - np.sin(phi) ** 2)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        r = (1 / k1z) + (mu_2 / k2z) * (m_minus / m_plus)
        return -r

    def b_mu(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Third coefficient in quadratic equation with respect to sqrt( epsilon ) for TE mode
        phi - angle of incidence
        a - thickness of HRIM
        e2 - phase change in SiO2
        epsilon_1 = 1 - dielectric coefficient of Air
        mu_1 = 1 - magnetic coefficient of Air
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_mu_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_mu_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        k1z = np.sqrt(0j + 1 - np.sin(phi) ** 2)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        return (1 / k1z) * (mu_2 / k2z) * (m_minus / m_plus)

    def c_mu(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3):
        """
        Fourth coefficient in quadratic equation with respect to sqrt( epsilon ) for TE mode
        phi - angle of incidence
        a - thickness of HRIM
        e2 - phase change in SiO2
        epsilon_1 = 1 - dielectric coefficient of Air
        mu_1 = 1 - magnetic coefficient of Air
        epsilon_2 - dielectric coefficient of SiO2
        mu_2 - magnetic coefficient of Si
        epsilon_3 - dielectric coefficient of Si
        mu_3 - magnetic coefficient of Si
        """
        m_plus = m_mu_plus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        m_minus = m_mu_minus(phi, e2, epsilon_2, mu_2, epsilon_3, mu_3)
        k1z = np.cos(phi)
        k2z = np.sqrt(0j + epsilon_2 * mu_2 - np.sin(phi) ** 2)
        return ((1 / k1z) + (mu_2 / k2z) * (m_minus / m_plus)) / (1j * a)

    def equation_coefficients(a1, b1, c1, a2, b2, c2):
        """
        Finding coefficients in fourth grade equation with respect to epsilon
        a1, b1, c1 - TM mode
        a2, b2, c2 - TE mode
        """
        A = (1 - b1 * b2) / (a1 - a2 * b1)
        A = A - a2 / 2
        B = (c1 - b1 * c2) / (a1 - a2 * b1)
        C = a2 * a2 / 4 - b2
        return [A * A - C, 2 * A * B + c2, B * B]

    e2 = exp_phase(b, phi, epsilon_2, mu_2)
    a1 = a_epsilon(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3)
    b1 = b_epsilon(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3)
    c1 = c_epsilon(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3)
    a2 = a_mu(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3)
    b2 = b_mu(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3)
    c2 = c_mu(phi, a, e2, epsilon_2, mu_2, epsilon_3, mu_3)

    coefficients1 = equation_coefficients(a1, b1, c1, a2, b2, c2)
    epsilons = np.roots(coefficients1)
    '''
    print(equation_coefficients(1 ,2 ,3 ,3 ,2 ,1))
    print(equation_coefficients(3,2,1,1,2,3))
    print(np.roots(equation_coefficients(1 ,2 ,3 ,3 ,2 ,1)))
    print(np.roots(equation_coefficients(3,2,1,1,2,3)))
    '''
    coefficients2 = equation_coefficients(a2, b2, c2, a1, b1, c1)
    mus = np.roots(coefficients2)
    # print((a2, b2, c2, a1, b1, c1))

    # eps = (b1 * c2 - c1) / ( 1 - b2)
    # mu = (b2 * c1 - c2) / ( 1 - b1)

    return [epsilons, mus]
    # return [eps, mu]


def solutions_graph(phi, a, b, epsilon_2, mu_2, epsilon_3, mu_3):
    """
    Unsuccesfull attempt to remake massive for 3d surface graph
    """
    # array = [[0] * len(phi)] * 4
    array = [[0] * len(phi)] * 8
    for j in range(len(phi)):
        x = solutions(phi[j], a, b, epsilon_2, mu_2, epsilon_3, mu_3)
        y = x[0]
        z = x[1]
        epsilons_real = [0] * 4
        epsilons_imag = [0] * 4
        mus_real = [0] * 4
        mus_imag = [0] * 4
        for i in range(4):
            array[i][j] = y[i].real
            array[i + 4][j] = y[i].imag

            epsilons_real[i] = y[i].real
            epsilons_imag[i] = y[i].imag
            mus_real[i] = z[i].real
            mus_imag[i] = z[i].imag
    # return [epsilons_real, epsilons_imag, mus_real, mus_imag]
    return array


def graph_angle():
    """
    Drawing graphs for epsilon and mu with respect to angle of incidence
    """
    # ax = plt.figure().add_subplot(projection='3d')
    for j in range(2):
        X = np.arange(0.1, np.pi / 2, 0.001)
        Y = [0] * len(X)
        Z = [0] * len(X)
        for i in range(len(X)):
            k = solutions(X[i], 0.1, 1, (1.46 ** 2), 1, (3.98 + 0.03j) ** 2, 1)[0]
            Y[i] = k[j].real
            Z[i] = k[j].imag

        plt.figure().add_subplot(projection="3d").scatter(X, Y, Z, "maroon")
        plt.xlabel("$Angle$ $of$ $incidence$")
        plt.ylabel("$Re[\epsilon]$")
        # plt.figure().add_subplot(projection='3d').set_zlabel("$Im[\epsilon]$")

    # ax.view_init(50, 40)
    plt.show()

    for j in range(2):
        X = np.arange(0.1, np.pi / 2, 0.001)
        Y = [0] * len(X)
        Z = [0] * len(X)
        for i in range(len(X)):
            k = solutions(X[i], 0.1, 1, (1.46 ** 2), 1, (3.98 + 0.03j) ** 2, 1)[1]
            Y[i] = k[j].real
            Z[i] = k[j].imag

        plt.figure().add_subplot(projection="3d").scatter(X, Y, Z, "maroon")
        plt.xlabel("$Angle$ $of$ $incidence$")
        plt.ylabel("$Re[\mu]$")
        # plt.figure().add_subplot(projection='3d').set_zlabel("$Im[\mu]$")

    # ax.view_init(50, 40)
    plt.show()


def epsilon_si(wave_lenght):
    """
    Finding complex epsilon for known wave lenght using to files with data for Si
    Using linear approximation
    """
    si_n = []
    with open("si_n.csv", newline="\n") as csvfile:
        ar = csv.reader(csvfile, delimiter=",")
        for row in ar:
            si_n.append(row)
    for i in si_n:
        i[0] = float(i[0])
        i[1] = float(i[1])

    si_k = []
    with open("si_k.csv", newline="\n") as csvfile:
        ar = csv.reader(csvfile, delimiter=",")
        for row in ar:
            si_k.append(row)
    for i in si_k:
        i[0] = float(i[0])
        i[1] = float(i[1])

    n = si_n[0][1]
    for iter in range(len(si_n) - 1):
        if si_n[iter][0] < wave_lenght and si_n[iter + 1][0] > wave_lenght:
            n = si_n[iter][1] + (si_n[iter + 1][1] - si_n[iter][1]) * (
                wave_lenght - si_n[iter][0]
            ) / (si_n[iter + 1][0] - si_n[iter][0])

    k = si_k[0][1]
    for iter in range(len(si_k) - 1):
        if si_k[iter][0] < wave_lenght and si_k[iter + 1][0] > wave_lenght:
            k = si_k[iter][1] + (si_k[iter + 1][1] - si_k[iter][1]) * (
                wave_lenght - si_k[iter][0]
            ) / (si_k[iter + 1][0] - si_k[iter][0])
    k = si_k[0][1]

    return (n + 1j * k) ** 2


def epsilon_sio2(wave_lenght):
    """
    Finding complex epsilon for known wave lenght using to files with data for SiO2
    Using linear approximation
    """
    si02_n = []
    with open("sio2_n.csv", newline="\n") as csvfile:
        ar = csv.reader(csvfile, delimiter=",")
        for row in ar:
            si02_n.append(row)
    for i in si02_n:
        i[0] = float(i[0])
        i[1] = float(i[1])

    n = si02_n[0][1]
    for iter in range(len(si02_n) - 1):
        if si02_n[iter][0] < wave_lenght and si02_n[iter + 1][0] > wave_lenght:
            n = si02_n[iter][1] + (si02_n[iter + 1][1] - si02_n[iter][1]) * (
                wave_lenght - si02_n[iter][0]
            ) / (si02_n[iter + 1][0] - si02_n[iter][0])

    return (n) ** 2


def graph_wave_lenght():
    """
    Drawing graphs for epsilon and mu with respect to wave_lenght
    """
    # ax = plt.figure().add_subplot(projection='3d')
    for j in range(2):
        X = np.arange(0.2, 0.8, 0.001)
        Y = [0] * len(X)
        Z = [0] * len(X)
        for i in range(len(X)):
            k = solutions(
                np.pi / 4, 0.1, 1, epsilon_sio2(X[i]), 1, epsilon_si(X[i]), 1
            )[0]
            Y[i] = k[j].real
            Z[i] = k[j].imag

        plt.figure().add_subplot(projection="3d").scatter(X, Y, Z, "maroon")
        plt.xlabel("$Wave$ $lenght$")
        plt.ylabel("$Re[\epsilon]$")
        # plt.figure().add_subplot(projection='3d').set_zlabel("$Im[\epsilon]$")

    # ax.view_init(50, 40)
    plt.show()

    for j in range(2):
        X = np.arange(0, 0.8, 0.001)
        Y = [0] * len(X)
        Z = [0] * len(X)
        for i in range(len(X)):
            k = solutions(
                np.pi / 4, 0.1, 1, epsilon_sio2(X[i]), 1, epsilon_si(X[i]), 1
            )[1]
            Y[i] = k[j].real
            Z[i] = k[j].imag

        plt.figure().add_subplot(projection="3d").scatter(X, Y, Z, "maroon")
        plt.xlabel("$Wave$ $lenght$")
        plt.ylabel("$Re[\mu]$")
        # plt.figure().add_subplot(projection='3d').set_zlabel("$Im[\mu]$")

    # ax.view_init(50, 40)
    plt.show()


def rho(phi, a, b, epsilon, mu, wave_lenght):
    """ """
    epsilon_2 = epsilon_sio2(wave_lenght)
    epsilon_3 = epsilon_si(wave_lenght)
    mu_3 = 1
    mu_2 = 1
    e2 = exp_phase(b, phi, epsilon_2, mu_2)
    #e1 = exp_phase(a, phi, epsilon, mu)
    e1 = 1j * a * np.sqrt(epsilon * mu)
    r_s = r_TM(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3)
    r_p = r_TE(phi, e1, e2, epsilon, mu, epsilon_2, mu_2, epsilon_3, mu_3)
    #print(abs(r_s), abs(r_p))
    #print(abs(e1))
    r = r_s / r_p
    psi = np.arctan(np.abs(r))
    delta = np.angle(r)
    #return [psi, delta]
    return e1
    #return [r_s , r_p]

a0 = 0.05
b0 = 33.4


print(solutions(np.pi / 4, a0, b0, epsilon_sio2(0.55), 1, epsilon_si(0.55), 1))


for i in range(2):
    for j in range(2):
        ans = solutions(np.pi / 4, a0, b0, epsilon_sio2(0.55), 1, epsilon_si(0.55), 1)
        rho(np.pi / 4, a0, b0, ans[0][i], ans[1][j], 0.55)
'''
for i in range(2):
    for j in range(2):
        ans = solutions(np.pi / 4, a0, b0, epsilon_sio2(0.55), 1, epsilon_si(0.55), 1)
        rho(np.pi / 4, a0, b0, ans[0][0], ans[1][1], 0.55)
        rho(np.pi / 4, a0, b0, ans[0][1], ans[1][0], 0.55)
'''
#X = np.arange(0.2, 1.5, 0.0001)
'''
lenght = np.arange(0.2 , 0.8, 0.01)
for leng in lenght:
    print("-")
    for jter in range(1, 100):
        thick2 = jter / 100
        for iter in range(1, 100):
            thick = iter / 1
            ans = solutions(np.pi / 2 - 0.1, thick2, thick, epsilon_sio2(leng), 1, epsilon_si(leng), 1)
            k = rho(np.pi / 2 - 0.1, thick2, thick, ans[0][i], ans[1][j], 0.55)
            if abs(k) < 0.05: print(abs(k))
            #if abs(k[0]) < 0.1 and abs(k[1]) < 0.1: print((thick2, thick, leng, k))
'''

# rho(np.pi / 4, 0.1, 1, ans[0], ans[1], 0.55)

#graph_angle()
#graph_wave_lenght()

(0.5, 33.4, 0.7853981633974483, (0.854496391500946-0.5212593796387454j))
