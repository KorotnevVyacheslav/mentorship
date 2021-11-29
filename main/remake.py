import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D



def epsilon_si(wave_lenght):
    """
    Finding complex epsilon for known wave lenght using to files with data for Si
    Using linear approximation
    """
    wave_lenght *= 1000
    si_n = []
    with open("si_n.csv", newline="\n") as csvfile:
        ar = csv.reader(csvfile, delimiter="	")
        for row in ar:
            si_n.append(row)
    for i in si_n:
        i[0] = float(i[0])
        i[1] = float(i[1])

    si_k = []
    with open("si_k.csv", newline="\n") as csvfile:
        ar = csv.reader(csvfile, delimiter="	")
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
    wave_lenght *= 1000
    si02_n = []
    with open("sio2_n.csv", newline="\n") as csvfile:
        ar = csv.reader(csvfile, delimiter="	")
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


def rho(phi, wave_lenght, a, b, epsilon1, mu1, epsilon, mu):

    epsilon2 = epsilon_sio2(wave_lenght)
    epsilon3 = epsilon_si(wave_lenght)
    mu2 = 1
    mu3 = 1

    k0 = 2 * np.pi / wave_lenght / 1000

    k1x = np.sqrt(0j + epsilon1 * mu1) * np.sin(phi)
    k1z = k0 * np.sqrt(0j + epsilon1 * mu1) * np.cos(phi)
    kz = k0 * np.sqrt(0j + epsilon * mu - k1x * k1x)
    k2z = k0 * np.sqrt(0j + epsilon2 * mu2 - k1x * k1x)
    k3z = k0 * np.sqrt(0j + epsilon3 * mu3 - k1x * k1x)

    if abs(np.exp(1j * kz)) > 1:
        kz = -kz
    if abs(np.exp(1j * k2z)) > 1:
        k2z = -k2z
    if abs(np.exp(1j * k3z)) > 1:
        k3z = -k3z

    e1 = np.exp(-2j * a * kz)
    e2 = np.exp(-2j * b * k2z)

    m_epsilon_plus = e2 + (k3z * epsilon2 - k2z * epsilon3) / (k3z * epsilon2 + k2z * epsilon3)
    m_epsilon_minus = 2 * e2 - m_epsilon_plus

    m_mu_plus = e2 + (k3z * mu2 - k2z * mu3) / (k3z * mu2 + k2z * mu3)
    m_mu_minus = 2 * e2 - m_mu_plus


    n_epsilon_plus = e1 + (k2z * epsilon * m_epsilon_plus - kz * epsilon2 * m_epsilon_minus) / (k2z * epsilon * m_epsilon_plus + kz * epsilon2 * m_epsilon_minus)
    n_epsilon_minus = 2 * e1 - n_epsilon_plus

    n_mu_plus = e1 + (k2z * mu * m_mu_plus - kz * mu2 * m_mu_minus) / (k2z * mu * m_mu_plus + kz * mu2 * m_mu_minus)
    n_mu_minus = 2 * e1 - n_mu_plus


    r_p = (kz * epsilon1 * n_epsilon_plus - k1z * epsilon * n_epsilon_minus) / (kz * epsilon1 * n_epsilon_plus + k1z * epsilon * n_epsilon_minus)
    r_s = (kz * mu1 * n_mu_plus - k1z * mu * n_mu_minus) / (kz * mu1 * n_mu_plus + k1z * mu * n_mu_minus)


    k = 0
    '''
    if abs(r_s) > 1 or abs(r_p) > 1:
        print(phi, wave_lenght, abs(r_s), abs(r_p))
        k = 1
    '''
    rho = r_p / r_s
    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)
    return [psi, delta, 0]


def solutions(phi, wave_lenght, a, b, epsilon1, mu1):

    epsilon2 = epsilon_sio2(wave_lenght)
    epsilon3 = epsilon_si(wave_lenght)
    mu2 = 1
    mu3 = 1

    k0 = 2 * np.pi / wave_lenght / 1000

    k1x = np.sqrt(0j + epsilon1 * mu1) * np.sin(phi)
    k1z = k0 * np.sqrt(0j + epsilon1 * mu1) * np.cos(phi)
    k2z = k0 * np.sqrt(0j + epsilon2 * mu2 - k1x * k1x)
    k3z = k0 * np.sqrt(0j + epsilon3 * mu3 - k1x * k1x)


    if abs(np.exp(1j * k2z)) > 1:
        k2z = -k2z
    if abs(np.exp(1j * k3z)) > 1:
        k3z = -k3z

    e2 = np.exp(-2j * b * k2z)

    m_epsilon_plus = e2 + (k3z * epsilon2 - k2z * epsilon3) / (k3z * epsilon2 + k2z * epsilon3)
    m_epsilon_minus = 2 * e2 - m_epsilon_plus

    m_mu_plus = e2 + (k3z * mu2 - k2z * mu3) / (k3z * mu2 + k2z * mu3)
    m_mu_minus = 2 * e2 - m_mu_plus


    a_epsilon = -k0 * (epsilon1 / k1z) + k0 * (epsilon2 / k2z) * (m_epsilon_minus / m_epsilon_plus)
    b_epsilon = -k0 * k0 * epsilon1 * epsilon2 * m_epsilon_minus / (k1z * k2z * m_epsilon_plus)
    c_epsilon = a_epsilon / (-1j * k0 * a)

    a_mu = k0 * ((mu2 / k2z) * (m_mu_minus / m_mu_plus) - mu1 / k1z)
    b_mu = -k0 * k0 * mu1 * mu2 * m_mu_minus / (k1z * k2z * m_mu_plus)
    c_mu = a_mu / (-1j * k0 * a)

    #if abs(np.exp(1j * k0 * epsilon * mu)) > 1:

    def equation_coefficients(a1, b1, c1, a2, b2, c2):
        """
        Finding coefficients in fourth grade equation with respect to epsilon
        a1, b1, c1 - TM mode
        a2, b2, c2 - TE mode
        """
        A = (1 - b1 * b2) / (a1 - a2 * b1) - a2 / 2
        B = (c1 - b1 * c2) / (a1 - a2 * b1)
        C = a2 * a2 / 4 - b2
        return [A * A - C, 2 * A * B + c2, B * B]


    coefficients1 = equation_coefficients(a_epsilon, b_epsilon, c_epsilon, a_mu, b_mu, c_mu)
    epsilons = [0] * 4
    epsilons[0] = np.roots(coefficients1)[0]
    epsilons[1] = np.roots(coefficients1)[1]

    mu_eps = [0] * 4
    for i in range(2):
        tmp1 = 1 / a_epsilon - b_mu / a_mu
        tmp2 = b_epsilon / a_epsilon - 1 / a_mu
        tmp3 = c_epsilon / a_epsilon - c_mu / a_mu
        mu_eps[i] = (- epsilons[i] * tmp1 - tmp3) / tmp2

    coefficients2 = equation_coefficients(a_mu, b_mu, c_mu, a_epsilon, b_epsilon, c_epsilon)
    mus = [0] * 4
    mus[0] = np.roots(coefficients2)[0]
    mus[1] = np.roots(coefficients2)[1]


    if (mus[0] == mu_eps[0] or mus[1] == mu_eps[1]) or (mus[1] == mu_eps[0] or mus[0] == mu_eps[1]):
        print("All is OKey")

    a_epsilon = - a_epsilon
    a_mu = - a_mu

    coefficients1 = equation_coefficients(a_epsilon, b_epsilon, c_epsilon, a_mu, b_mu, c_mu)
    epsilons2 = np.roots(coefficients1)

    mu_eps2 = [0] * 2
    for i in range(2):
        tmp1 = 1 / a_epsilon - b_mu / a_mu
        tmp2 = b_epsilon / a_epsilon - 1 / a_mu
        tmp3 = c_epsilon / a_epsilon - c_mu / a_mu
        mu_eps2[i] = (- epsilons2[i] * tmp1 - tmp3) / tmp2

    coefficients2 = equation_coefficients(a_mu, b_mu, c_mu, a_epsilon, b_epsilon, c_epsilon)
    mus2 = np.roots(coefficients2)

    mus[2] = mus2[0]
    mus[3] = mus2[1]
    epsilons[2] = epsilons2[0]
    epsilons[3] = epsilons2[1]
    mu_eps[2] = mu_eps2[0]
    mu_eps[3] = mu_eps2[1]

    #print(mus, mu_eps)

    return [epsilons, mus]


def main():
    N = 300
    a0 = 5.1
    b0 = 285
    a_ar = np.linspace(200, 250, 1)
    b_ar = np.linspace(1, 1, 1)

    num = 0
    eps1  = 1
    mu1 = 1

    for koef1 in a_ar:
        for koef2 in b_ar:
            b0 = 220
            for iter in range(1):
                iter = 3
                epsilon0 = solutions(np.pi / 4, 0.55, a0, b0, eps1, mu1)[0][iter]
                mu0 = solutions(np.pi / 4, 0.55, a0, b0, eps1, mu1)[1][iter]
                #epsilon0 = 24 - 11j
                print(eps1,a0,b0,epsilon0, mu0)

                #mu0 = 1
                #epsilon0 = 21.2777 + 26.9458j

                #epsilon0 = -15.243 +  0.40284j
                #print(np.sqrt(epsilon0))
                #epsilon0 = epsilon0 * koef1
                #mu0 = mu0 * (koef2)
                #print(eps1, epsilon0, mu0, iter)

                if  epsilon0.imag > 0 or mu0.imag > 0 or (abs(np.exp(1j * epsilon0 * mu0)) > 1 and iter < 2):
                    print("passed ", iter)
                    continue
                else:
                    num +=1

                angle = np.linspace(0.5, np.pi / 2 - 0.01, N)
                wave = np.linspace(0.4, 0.6, N)
                '''
                with open("tables/angle.csv", "wt") as fp:
                    writer = csv.writer(fp, delimiter=" ")
                    writer.writerow(angle)

                with open("tables/wave.csv", "wt") as fp:
                    writer = csv.writer(fp, delimiter=" ")
                    writer.writerow(wave)
                '''
                XX, YY = np.meshgrid(angle, wave)
                delt = XX + YY
                delt2 = XX + YY
                for i in range(N):
                    for j in range(N):
                        k = rho(angle[j], wave[i], a0, b0, eps1, mu1, epsilon0, mu0)
                        delt[i][j] = k[1]
                        delt2[i][j] = k[0]
                        num += k[2]
                #print(type(delt))
                #print(num)
                ZZ = delt
                fig, ax = plt.subplots(1, 2)



                pcm = ax[0].pcolor(YY, XX, ZZ, cmap="jet")
                fig.colorbar(pcm, ax=ax[0], extend="max")
                plt.ylabel("$Angle$ $of$ $incidence$")
                plt.xlabel("$\lambda$ $,$ $nm$")


                ZZ = delt2
                pcm = ax[1].pcolor(YY, XX, ZZ, cmap="jet")
                fig.colorbar(pcm, ax=ax[1], extend="max")
                plt.ylabel("$Angle$ $of$ $incidence$")
                plt.xlabel("$\lambda$ $,$ $nm$")
                '''
                if iter == 2:
                    plt.savefig("a_" + str(num))
                if iter == 3:
                    plt.savefig("b_" + str(num))
                '''
    plt.show()

main()
