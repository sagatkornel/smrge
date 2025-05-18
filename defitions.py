import numpy as np

# Physical constants
alpha_EM = 1/127.937      # Electromagnetic coupling constant
alpha_S  = 0.1184          # Strong coupling constant

# Weak mixing angle
sin2_theta = 0.23126       # sin^2(theta_W)
cos2_theta = 1 - sin2_theta
sin_theta  = np.sqrt(sin2_theta)
cos_theta  = np.sqrt(cos2_theta)

# Particle masses (in GeV)
M_t = 173.1    # Top quark mass
M_h = 125.5    # Higgs boson mass
M_Z = 91.1876  # Z boson mass

# Uncertainties (errors)
delta_alpha_EN   = 1/0.015   # EM coupling uncertainty
delta_alpha_S    = 0.0007    # Strong coupling uncertainty
delta_sin2_theta = 0.00005   # Weak mixing angle uncertainty
# Propagated uncertainties
delta_cos2_theta = 1 - (sin2_theta + delta_sin2_theta)
delta_sin_theta = np.sqrt(sin2_theta + delta_sin2_theta) - sin_theta
delta_cos_theta = np.sqrt(cos2_theta + delta_cos2_theta) - cos_theta

delta_M_t = 0.7   # GeV
delta_M_h = 0.5   # GeV

# Initial gauge couplings at scale M_Z
g1_initial = np.sqrt(5/3) * np.sqrt(4*np.pi*alpha_EM) / cos_theta
g2_initial = np.sqrt(4*np.pi*alpha_EM) / sin_theta
g3_initial = (
    1.1645
    + 0.0031 * ((alpha_S - 0.1184) / 0.0007)
    - 0.00046 * (M_t - 173.15)
)

# Yukawa couplings at scale M_Z
y_t_initial = (
    0.93587
    + 0.00557 * (M_t - 173.15)
    - 0.00003 * (M_h - 125)
    - 0.00041 * ((alpha_S - 0.1184) / 0.0007)
)
delta_y_t_initial = 0.00200

y_b_initial   = 0.016
y_tau_initial = 0.0102

# Higgs self-coupling at scale M_Z
lambda_initial         = (
    0.12577
    + 0.00205 * (M_h - 125)
    - 0.00004 * (M_t - 173.15)
)
delta_lambda_initial = 0.00140

# Central initial state vectors for solver
y0 = [
    g1_initial,
    g2_initial,
    g3_initial,
    y_t_initial,
    y_b_initial,
    y_tau_initial,
    lambda_initial,
]
# Without bottom and tau Yukawas (e.g., for 1-loop toy runs)
y1 = [g1_initial, g2_initial, g3_initial, y_t_initial, lambda_initial]

# Energy-scale parameters (logarithmic t = ln(mu/M_Z))
mu_min = M_t / M_Z
mu_max = 1e21 / M_Z  # ~10^21 GeV / M_Z

t_min = np.log(mu_min)
t_max = np.log(mu_max)

print("asd")