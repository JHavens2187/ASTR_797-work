__author__ = "Joseph Havens"
__date__ = "2024-10-03"
__version__ = "no-library"
__complextiy__ = "O(n^2) + O(Z * N)"  # due to root finding for Legendre polynomials and manual math functions

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science"])

# =============== constants ===============
Om = 0.3
Ol = 0.7
h = 0.7

pi = 3.141592653589793
n = 5
c = 3e5  # speed of light in km/s
H_0 = 100 * h


# =============== manual math functions ===============
def diy_factorial(n):
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def cos(x, terms=10):
    """Taylor series expansion for cos(x)"""
    # Normalize x to be within -2pi and 2pi for better precision
    x = x % (2 * pi)

    cos_x = 0
    for n in range(terms):
        numerator = (-1) ** n * x ** (2 * n)
        denominator = diy_factorial(2 * n)
        cos_x += numerator / denominator
    return cos_x


def sqrt(x, iterations=10):
    """Babylonian method for sqrt(x)"""
    if x < 0:
        raise ValueError("Cannot sqrt negative numbers")
    if x == 0:
        return 0
    guess = x / 2.0
    for _ in range(iterations):
        guess = 0.5 * (guess + x / guess)
    return guess


# ================ Core Legendre Logic (Pure Python) =================
def get_legendre_roots_and_weights(n):
    """Computes roots and weights using only the math module."""
    roots = [0] * n
    weights = [0] * n

    for i in range(1, n + 1):
        # Initial guess for the root
        x = cos(pi * (i - 0.25) / (n + 0.5))

        # Newton's Method
        while True:
            p_prev, p_curr = 0.0, 1.0
            for k in range(n):
                p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
                p_prev, p_curr = p_curr, p_next

            # Derivative of P_n at x
            dp = n * (x * p_curr - p_prev) / (x**2 - 1)

            x_new = x - p_curr / dp
            if abs(x_new - x) < 1e-15:
                break
            x = x_new

        roots[i - 1] = x
        weights[i - 1] = 2 / ((1 - x**2) * (dp**2))

    return roots, weights


# Calculate them once at the start
nodes, weight_vals = get_legendre_roots_and_weights(n)


# ================ Math Functions =================
def E(z):
    return sqrt(Om * (1 + z) ** 3 + Ol)


def gaussian_quadrature(a, b, f):
    integral = 0
    for i in range(n):
        # Rescale node from [-1, 1] to [a, b]
        xi = 0.5 * (b - a) * nodes[i] + 0.5 * (a + b)
        integral += weight_vals[i] * f(xi)
    return integral * 0.5 * (b - a)


def d_A(z):
    if z == 0:
        return 0
    integral = gaussian_quadrature(0, z, lambda x: 1 / E(x))
    return (c / (H_0 * (1 + z))) * integral


def theta_eff(z, galaxy_type):
    if z == 0:
        return 0
    # Galaxy size logic
    b = -0.75 if galaxy_type == "SF" else -1.48
    A = 8.9 if galaxy_type == "SF" else 5.6
    R = (A * (1 + z) ** b) / 1000  # kpc to Mpc

    dist = d_A(z)
    theta_rad = R / dist
    return theta_rad * (180 / pi) * 3600  # convert to arcseconds


# ================= Plotting =================
def plot_results():
    # Create z_values without np.linspace
    num_points = 100
    z_max = 3.0
    z_values = [i * (z_max / (num_points - 1)) for i in range(num_points)]

    da_vals = [d_A(z) for z in z_values]
    sf_vals = [theta_eff(z, "SF") for z in z_values]
    pas_vals = [theta_eff(z, "Passive") for z in z_values]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(z_values, da_vals, label="Angular Diameter Distance", color="blue")
    ax2.plot(z_values, sf_vals, label="Star-Forming", color="orange")
    ax2.plot(z_values, pas_vals, label="Passive", color="green")

    ax1.set_xlabel("Redshift z")
    ax1.set_ylabel(r"$d_A(z)$ [Mpc]")
    ax2.set_ylabel(r"$\theta_{\mathrm{eff}}$ [arcsec]")
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.75))
    plt.show()


if __name__ == "__main__":
    plot_results()
