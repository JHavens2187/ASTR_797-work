__author__ = "Joseph Havens"
__date__ = "2024-10-03"
__version__ = "numpy"
__complextiy__ = "O(n)"  # leveraging numpy's optimized routines

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])  # because pretty plots are paramount!
# I WISH I could've avoided using numpy as well, but this code would've been twice as long
import numpy as np

# =============== constants ===============
Om = 0.3
Ol = 0.7
h = 0.7

n = 5  # order of the Legendre polynomial
c = 3e5  # speed of light in km/s
H_0 = 100 * h  # Hubble constant in km/s/Mpc
Omega = 0.25 * (np.pi / 180) ** 2  # solid angle in deg^2 converted to steradians
# ==========================================

# get the nodes of the nth order legendre polynomial - tragically we couldn't (shouldn't?) compute the Legendre polynomial roots ourselves
# yes we technically get the weights here too... but thats not as fun
nodes, _ = np.polynomial.legendre.leggauss(n)

# get the derivative of the nth order Legendre polynomial at the nodes
Legendre = np.polynomial.Legendre.basis(n).deriv()


# ================ math functions =================
# main math functions
def gaussian_quadratures(a, b, f, w, zeta):
    integral = 0
    for i in range(n):
        xi = 0.5 * (b - a) * zeta[i] + 0.5 * (a + b)
        integral += w[i] * f(xi)
    integral *= 0.5 * (b - a)
    return integral


def weights(zeta, n):
    w = []
    for i in range(n):
        P_prime = Legendre(zeta[i])
        w.append(2 / ((1 - zeta[i] ** 2) * (P_prime**2)))
    return w


def E(z):
    return np.sqrt(Om * (1 + z) ** 3 + Ol)


def d_A(z):
    return (c / (H_0 * (1 + z))) * gaussian_quadratures(
        0, z, lambda x: 1 / E(x), weights(nodes, n), nodes
    )


def R_eff(z, galaxy_type):
    if galaxy_type == "SF":
        A = 8.9  # kpc
        b = -0.75
        return A * (1 + z) ** b / 1000  # convert to Mpc
    elif galaxy_type == "Passive":
        A = 5.6  # kpc
        b = -1.48
        return A * (1 + z) ** b / 1000  # convert to Mpc
    else:
        raise ValueError("Unknown galaxy type. Use 'SF' or 'Passive'.")


def theta_eff(z, galaxy_type):
    R = R_eff(z, galaxy_type)  # in Mpc
    D_A = d_A(z)  # in Mpc
    theta = R / D_A  # in radians
    return theta * (180 / np.pi) * 3600  # convert to arcseconds


def Volume_Comoving(z1, z2, Omega):
    # dV_c = D_H * ((1 + z)^2 * d_A^2) / E(z)) dz dOmega
    d_A_func = lambda z: d_A(z)
    integrand = lambda z: ((1 + z) ** 2 * d_A_func(z) ** 2) / E(z)
    integral = gaussian_quadratures(z1, z2, integrand, weights(nodes, n), nodes)
    D_H = c / H_0
    return D_H * integral * Omega  # in Mpc^3


# ================= plotting functions =================
def plot_d_A_and_theta():
    z_values = np.linspace(0, 3, 100)
    d_A_values = [d_A(z) for z in z_values]
    theta_SF_values = [theta_eff(z, "SF") for z in z_values]
    theta_Passive_values = [theta_eff(z, "Passive") for z in z_values]
    # print(d_A_values)  # oh god thats a lot of numbers!
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(z_values, d_A_values, label="Angular Diameter Distance", color="blue")
    ax2.plot(z_values, theta_SF_values, label="Star-Forming Galaxy", color="orange")
    ax2.plot(z_values, theta_Passive_values, label="Passive Galaxy", color="green")
    ax1.set_xlabel("Redshift z")
    ax1.set_ylabel(r"Angular Diameter Distance $d_A(z)$ [Mpc]")
    ax2.set_ylabel(r"Effective Angular Size $\theta_{\mathrm{eff}}(z)$ [arcseconds]")
    ax2.set_xlim(min(z_values), max(z_values))
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.75))
    fig.suptitle("Angular Diameter Distance and Effective Angular Size vs Redshift")
    fig.savefig("dA_and_theta_vs_z.pdf")
    plt.show()


# ================= Driver =================
if __name__ == "__main__":
    print(
        rf"Comoving Volume for $0.05 \leq z \leq 1$: {round(Volume_Comoving(0.05, 1, Omega), 2)} $Mpc^3$"
    )
    print(
        rf"Comoving Volume for $1 \leq z \leq 2$: {round(Volume_Comoving(1, 2, Omega), 2)} $Mpc^3$"
    )
    print(
        rf"Comoving Volume for $2 \leq z \leq 3$: {round(Volume_Comoving(2, 3, Omega), 2)} $Mpc^3$"
    )

    plot_d_A_and_theta()
