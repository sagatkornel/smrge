import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import defitions as defs


def beta_nobtau_1loop(t, y):
    """
    1-loop beta functions without bottom and tau Yukawas:
    state y = [g1, g2, g3, yt, lambda]
    """
    g1, g2, g3, yt, la = y

    # 1-loop gauge
    beta_g1 = 41/10 * g1**3
    beta_g2 = -19/6 * g2**3
    beta_g3 = -7 * g3**3

    # 1-loop top Yukawa
    beta_yt = (3 * yt**3) / 2 - (yt * (17 * g1**2 + 45 * g2**2 + 160 * g3**2 - 60 * yt**2)) / 20

    # 1-loop Higgs self-coupling
    beta_la = (
        (27 * g1**4) / 200
        + (9 * g1**2 * g2**2) / 20
        + (9 * g2**4) / 8
        - (9 * g1**2 * la) / 5
        - 9 * g2**2 * la
        + 24 * la**2
        + 12 * la * yt**2
        - 6 * yt**4
    )

    # normalize by (4*pi)^2
    dg1 = beta_g1 / (4 * np.pi)**2
    dg2 = beta_g2 / (4 * np.pi)**2
    dg3 = beta_g3 / (4 * np.pi)**2
    dyt = beta_yt / (4 * np.pi)**2
    dla = beta_la / (4 * np.pi)**2

    return [dg1, dg2, dg3, dyt, dla]


def beta_nobtau_2loop(t, y):
    """
    2-loop beta functions without bottom and tau Yukawas:
    state y = [g1, g2, g3, yt, lambda]
    """
    g1, g2, g3, yt, la = y

    # 1- and 2-loop gauge
    beta_g1_1 = 41/10 * g1**3
    beta_g1_2 = (199 * g1**5 + 5 * g1**3 * (27 * g2**2 + 88 * g3**2 - 17 * yt**2)) / 50

    beta_g2_1 = -19/6 * g2**3
    beta_g2_2 = (g2**3 * (27 * g1**2 + 5 * (35 * g2**2 + 72 * g3**2 - 9 * yt**2))) / 30

    beta_g3_1 = -7 * g3**3
    beta_g3_2 = (g3**3 * (11 * g1**2 + 45 * g2**2 - 20 * (13 * g3**2 + yt**2))) / 10

    # 1- and 2-loop top Yukawa
    beta_yt_1 = (3 * yt**3) / 2 - (yt * (17 * g1**2 + 45 * g2**2 + 160 * g3**2 - 60 * yt**2)) / 20
    beta_yt_2 = (
        yt * (
            2374 * g1**4
            + 5 * g1**2 * (-108 * g2**2 + 304 * g3**2 + 1179 * yt**2)
            - 75 * (
                92 * g2**4
                - 3 * g2**2 * (48 * g3**2 + 75 * yt**2)
                + 4 * (
                    432 * g3**4
                    - 24 * la**2
                    + 48 * yt**2 * (la + yt**2)
                    - 16 * g3**2 * (9 * yt**2)
                    + 9 * yt**4
                )
            )
        )
    ) / 1200

    # 1- and 2-loop Higgs self-coupling
    beta_la_1 = (
        (27 * g1**4) / 200
        + (9 * g1**2 * g2**2) / 20
        + (9 * g2**4) / 8
        - (9 * g1**2 * la) / 5
        - 9 * g2**2 * la
        + 24 * la**2
        + 12 * la * yt**2
        - 6 * yt**4
    )
    beta_la_2 = (
        (-3411 * g1**6) / 2000
        + (305 * g2**6) / 16
        - 312 * la**3
        + 80 * g3**2 * la * yt**2
        - 144 * la**2 * yt**2
        - 32 * g3**2 * yt**4
        - 3 * la * yt**4
        + 30 * yt**6
        - (3 * g1**4 * (559 * g2**2 - 1258 * la + 228 * yt**2)) / 400
        + (3 * g2**2 * la * (72 * la + 5 * (3 * yt**2))) / 2
        - (g2**4 * (73 * la + 6 * (3 * yt**2))) / 8
        + (
            g1**2 * (
                -289 * g2**4
                + 12 * g2**2 * (39 * la + 42 * yt**2)
                + 8 * (
                    216 * la**2
                    + 5 * la * (17 * yt**2)
                    + 8 * (-2 * yt**4)
                )
            )
        ) / 80
    )

    dg1 = beta_g1_1 / (4 * np.pi)**2 + beta_g1_2 / (4 * np.pi)**4
    dg2 = beta_g2_1 / (4 * np.pi)**2 + beta_g2_2 / (4 * np.pi)**4
    dg3 = beta_g3_1 / (4 * np.pi)**2 + beta_g3_2 / (4 * np.pi)**4
    dyt = beta_yt_1 / (4 * np.pi)**2 + beta_yt_2 / (4 * np.pi)**4
    dla = beta_la_1 / (4 * np.pi)**2 + beta_la_2 / (4 * np.pi)**4

    return [dg1, dg2, dg3, dyt, dla]


def beta_full_1loop(t, y):
    """
    1-loop beta functions with bottom and tau Yukawas:
    state y = [g1, g2, g3, yt, yb, ytau, lambda]
    """
    g1, g2, g3, yt, yb, ytau, la = y
    
    # 1 and 2 loop Gauge couplings
    beta_g1_1 = 41/10*g1**3
    beta_g1_2 = (199 * g1**5 + 5 * g1**3 * (27 * g2**2 + 88 * g3**2 - 5 * yb**2 - 17 * yt**2 - 15 * ytau**2)) / 50
    
    beta_g2_1 = -19/6*g2**3
    beta_g2_2 = (g2**3 * (27 * g1**2 + 5 * (35 * g2**2 + 72 * g3**2 - 3 * (3 * (yb**2 + yt**2) + ytau**2)))) / 30
    
    beta_g3_1 = -7*g3**3
    beta_g3_2 = (g3**3 * (11 * g1**2 + 45 * g2**2 - 20 * (13 * g3**2 + yb**2 + yt**2))) / 10
    
    # 1 and 2 loop Yukawa couplings
    beta_yt_1 = (3 * (-yb**2 * yt + yt**3)) / 2 - (yt * (17 * g1**2 + 45 * g2**2 + 160 * g3**2 - 60 * yb**2 - 60 * yt**2 - 20 * ytau**2)) / 20
    beta_yt_2 = (yt * (2374 * g1**4 + 5 * g1**2 * (-108 * g2**2 + 304 * g3**2 + 21 * yb**2 + 1179 * yt**2 + 450 * ytau**2)
                 - 75 * (92 * g2**4 - 3 * g2**2 * (48 * g3**2 + 33 * yb**2 + 75 * yt**2 + 10 * ytau**2)
                         + 4 * (432 * g3**4 - 24 * la**2 + yb**4 + 48 * yt**2 * (la + yt**2)
                                - 16 * g3**2 * (yb**2 + 9 * yt**2) + 9 * yt**2 * ytau**2
                                + 9 * ytau**4 + yb**2 * (11 * yt**2 - 5 * ytau**2))))) / 1200

    
    beta_yb_1 = (3 * (yb**3 - yb * yt**2)) / 2 - (yb * (5 * g1**2 + 45 * g2**2 + 160 * g3**2 - 60 * yb**2 - 60 * yt**2 - 20 * ytau**2)) / 20
    beta_yb_2 = (yb * (-254*g1**4 
                  + 5*g1**2 * (-324*g2**2 + 496*g3**2 + 711*yb**2 + 273*yt**2 + 450*ytau**2)
                  - 75 * (92*g2**4 
                          - 3*g2**2 * (48*g3**2 + 75*yb**2 + 33*yt**2 + 10*ytau**2)
                          + 4 * (432*g3**4 - 24*la**2 + 48*la*yb**2 + 48*yb**4 
                                 + 11*yb**2*yt**2 + yt**4 
                                 - 16*g3**2 * (9*yb**2 + yt**2) 
                                 + (9*yb**2 - 5*yt**2)*ytau**2 
                                 + 9*ytau**4)))) / 1200
    
    beta_ytau_1 = (3 * ytau**3) / 2 - (ytau * (45*g1**2 + 45*g2**2 - 60*yb**2 - 60*yt**2 - 20*ytau**2)) / 20
    beta_ytau_2 = (
        ytau * (
            2742*g1**4
            + 5*g1**2 * (108*g2**2 + 50*yb**2 + 170*yt**2 + 537*ytau**2)
            + 25 * (
                -92*g2**4
                + 96*la**2
                + 15*g2**2 * (6*(yb**2 + yt**2) + 11*ytau**2)
                + 4 * (
                    80*g3**2 * (yb**2 + yt**2)
                    - 3 * (
                        9*yb**4
                        - 2*yb**2*yt**2
                        + 9*yt**4
                        + (16*la + 9*(yb**2 + yt**2)) * ytau**2
                        + 4*ytau**4
                    )
                )
            )
        )
    ) / 400
    
    # 1 and 2 loop scalar couplings
    beta_lam_1 = (
        (27 * g1**4) / 200
        + (9 * g1**2 * g2**2) / 20
        + (9 * g2**4) / 8
        - (9 * g1**2 * la) / 5
        - 9 * g2**2 * la
        + 24 * la**2
        + 12 * la * yb**2
        - 6 * yb**4
        + 12 * la * yt**2
        - 6 * yt**4
        + 4 * la * ytau**2
        - 2 * ytau**4
    )
    beta_lam_2 = (
        (-3411 * g1**6) / 2000
        + (305 * g2**6) / 16
        - 312 * la**3
        + 80 * g3**2 * la * yb**2
        - 144 * la**2 * yb**2
        - 32 * g3**2 * yb**4
        - 3 * la * yb**4
        + 30 * yb**6
        + 80 * g3**2 * la * yt**2
        - 144 * la**2 * yt**2
        - 42 * la * yb**2 * yt**2
        - 6 * yb**4 * yt**2
        - 32 * g3**2 * yt**4
        - 3 * la * yt**4
        - 6 * yb**2 * yt**4
        + 30 * yt**6
        - 48 * la**2 * ytau**2
        - la * ytau**4
        + 10 * ytau**6
        - (
            3 * g1**4
            * (559 * g2**2 - 1258 * la - 60 * yb**2 + 228 * yt**2 + 300 * ytau**2)
        ) / 400
        + (
            3 * g2**2
            * la
            * (72 * la + 5 * (3 * (yb**2 + yt**2) + ytau**2))
        ) / 2
        - (
            g2**4
            * (73 * la + 6 * (3 * (yb**2 + yt**2) + ytau**2))
        ) / 8
        + (
            g1**2
            * (
                -289 * g2**4
                + 12 * g2**2 * (39 * la + 18 * yb**2 + 42 * yt**2 + 22 * ytau**2)
                + 8
                * (
                    216 * la**2
                    + 5 * la * (5 * yb**2 + 17 * yt**2 + 15 * ytau**2)
                    + 8
                    * (
                        yb**4
                        - 2 * yt**4
                        - 3 * ytau**4
                    )
                )
            )
        ) / 80
    )
    
    # Differential equations
    dg1_dt = beta_g1_1 / (4*np.pi)**2 #+ beta_g1_2 / (4*np.pi)**4
    dg2_dt = beta_g2_1 / (4*np.pi)**2 #+ beta_g2_2 / (4*np.pi)**4
    dg3_dt = beta_g3_1 / (4*np.pi)**2 #+ beta_g3_2 / (4*np.pi)**4
    dyt_dt = beta_yt_1 / (4*np.pi)**2 #+ beta_yt_2 / (4*np.pi)**4
    dyb_dt = beta_yb_1 / (4*np.pi)**2 #+ beta_yb_2 / (4*np.pi)**4
    dytau_dt = beta_ytau_1 / (4*np.pi)**2 #+ beta_ytau_2 / (4*np.pi)**4
    dlam_dt = beta_lam_1 / (4*np.pi)**2 #+ beta_lam_2 / (4*np.pi)**4
    
    return [dg1_dt, dg2_dt, dg3_dt, dyt_dt, dyb_dt, dytau_dt, dlam_dt]


def beta_full_2loop(t, y):
    """
    2-loop beta functions with bottom and tau Yukawas:
    state y = [g1, g2, g3, yt, yb, ytau, lambda]
    """
    g1, g2, g3, yt, yb, ytau, la = y
    
    # 1 and 2 loop Gauge couplings
    beta_g1_1 = 41/10*g1**3
    beta_g1_2 = (199 * g1**5 + 5 * g1**3 * (27 * g2**2 + 88 * g3**2 - 5 * yb**2 - 17 * yt**2 - 15 * ytau**2)) / 50
    
    beta_g2_1 = -19/6*g2**3
    beta_g2_2 = (g2**3 * (27 * g1**2 + 5 * (35 * g2**2 + 72 * g3**2 - 3 * (3 * (yb**2 + yt**2) + ytau**2)))) / 30
    
    beta_g3_1 = -7*g3**3
    beta_g3_2 = (g3**3 * (11 * g1**2 + 45 * g2**2 - 20 * (13 * g3**2 + yb**2 + yt**2))) / 10
    
    # 1 and 2 loop Yukawa couplings
    beta_yt_1 = (3 * (-yb**2 * yt + yt**3)) / 2 - (yt * (17 * g1**2 + 45 * g2**2 + 160 * g3**2 - 60 * yb**2 - 60 * yt**2 - 20 * ytau**2)) / 20
    beta_yt_2 = (yt * (2374 * g1**4 + 5 * g1**2 * (-108 * g2**2 + 304 * g3**2 + 21 * yb**2 + 1179 * yt**2 + 450 * ytau**2)
                 - 75 * (92 * g2**4 - 3 * g2**2 * (48 * g3**2 + 33 * yb**2 + 75 * yt**2 + 10 * ytau**2)
                         + 4 * (432 * g3**4 - 24 * la**2 + yb**4 + 48 * yt**2 * (la + yt**2)
                                - 16 * g3**2 * (yb**2 + 9 * yt**2) + 9 * yt**2 * ytau**2
                                + 9 * ytau**4 + yb**2 * (11 * yt**2 - 5 * ytau**2))))) / 1200

    
    beta_yb_1 = (3 * (yb**3 - yb * yt**2)) / 2 - (yb * (5 * g1**2 + 45 * g2**2 + 160 * g3**2 - 60 * yb**2 - 60 * yt**2 - 20 * ytau**2)) / 20
    beta_yb_2 = (yb * (-254*g1**4 
                  + 5*g1**2 * (-324*g2**2 + 496*g3**2 + 711*yb**2 + 273*yt**2 + 450*ytau**2)
                  - 75 * (92*g2**4 
                          - 3*g2**2 * (48*g3**2 + 75*yb**2 + 33*yt**2 + 10*ytau**2)
                          + 4 * (432*g3**4 - 24*la**2 + 48*la*yb**2 + 48*yb**4 
                                 + 11*yb**2*yt**2 + yt**4 
                                 - 16*g3**2 * (9*yb**2 + yt**2) 
                                 + (9*yb**2 - 5*yt**2)*ytau**2 
                                 + 9*ytau**4)))) / 1200
    
    beta_ytau_1 = (3 * ytau**3) / 2 - (ytau * (45*g1**2 + 45*g2**2 - 60*yb**2 - 60*yt**2 - 20*ytau**2)) / 20
    beta_ytau_2 = (
        ytau * (
            2742*g1**4
            + 5*g1**2 * (108*g2**2 + 50*yb**2 + 170*yt**2 + 537*ytau**2)
            + 25 * (
                -92*g2**4
                + 96*la**2
                + 15*g2**2 * (6*(yb**2 + yt**2) + 11*ytau**2)
                + 4 * (
                    80*g3**2 * (yb**2 + yt**2)
                    - 3 * (
                        9*yb**4
                        - 2*yb**2*yt**2
                        + 9*yt**4
                        + (16*la + 9*(yb**2 + yt**2)) * ytau**2
                        + 4*ytau**4
                    )
                )
            )
        )
    ) / 400
    
    # 1 and 2 loop scalar couplings
    beta_lam_1 = (
        (27 * g1**4) / 200
        + (9 * g1**2 * g2**2) / 20
        + (9 * g2**4) / 8
        - (9 * g1**2 * la) / 5
        - 9 * g2**2 * la
        + 24 * la**2
        + 12 * la * yb**2
        - 6 * yb**4
        + 12 * la * yt**2
        - 6 * yt**4
        + 4 * la * ytau**2
        - 2 * ytau**4
    )
    beta_lam_2 = (
        (-3411 * g1**6) / 2000
        + (305 * g2**6) / 16
        - 312 * la**3
        + 80 * g3**2 * la * yb**2
        - 144 * la**2 * yb**2
        - 32 * g3**2 * yb**4
        - 3 * la * yb**4
        + 30 * yb**6
        + 80 * g3**2 * la * yt**2
        - 144 * la**2 * yt**2
        - 42 * la * yb**2 * yt**2
        - 6 * yb**4 * yt**2
        - 32 * g3**2 * yt**4
        - 3 * la * yt**4
        - 6 * yb**2 * yt**4
        + 30 * yt**6
        - 48 * la**2 * ytau**2
        - la * ytau**4
        + 10 * ytau**6
        - (
            3 * g1**4
            * (559 * g2**2 - 1258 * la - 60 * yb**2 + 228 * yt**2 + 300 * ytau**2)
        ) / 400
        + (
            3 * g2**2
            * la
            * (72 * la + 5 * (3 * (yb**2 + yt**2) + ytau**2))
        ) / 2
        - (
            g2**4
            * (73 * la + 6 * (3 * (yb**2 + yt**2) + ytau**2))
        ) / 8
        + (
            g1**2
            * (
                -289 * g2**4
                + 12 * g2**2 * (39 * la + 18 * yb**2 + 42 * yt**2 + 22 * ytau**2)
                + 8
                * (
                    216 * la**2
                    + 5 * la * (5 * yb**2 + 17 * yt**2 + 15 * ytau**2)
                    + 8
                    * (
                        yb**4
                        - 2 * yt**4
                        - 3 * ytau**4
                    )
                )
            )
        ) / 80
    )
    
    # Differential equations
    dg1_dt = beta_g1_1 / (4*np.pi)**2 + beta_g1_2 / (4*np.pi)**4
    dg2_dt = beta_g2_1 / (4*np.pi)**2 + beta_g2_2 / (4*np.pi)**4
    dg3_dt = beta_g3_1 / (4*np.pi)**2 + beta_g3_2 / (4*np.pi)**4
    dyt_dt = beta_yt_1 / (4*np.pi)**2 + beta_yt_2 / (4*np.pi)**4
    dyb_dt = beta_yb_1 / (4*np.pi)**2 + beta_yb_2 / (4*np.pi)**4
    dytau_dt = beta_ytau_1 / (4*np.pi)**2 + beta_ytau_2 / (4*np.pi)**4
    dlam_dt = beta_lam_1 / (4*np.pi)**2 + beta_lam_2 / (4*np.pi)**4
    
    return [dg1_dt, dg2_dt, dg3_dt, dyt_dt, dyb_dt, dytau_dt, dlam_dt]


class RGERunner:
    def __init__(self, loops=1, include_btau=False):
        """
        loops: 1 or 2
        include_btau: whether to include bottom and tau Yukawas
        """
        self.loops = loops
        self.include_btau = include_btau
        # initial state vector
        self.y0 = np.array(defs.y0 if include_btau else defs.y1)
        self.t_span = (defs.t_min, defs.t_max)

    def beta(self, t, y):
        if self.include_btau:
            if self.loops == 1:
                return beta_full_1loop(t, y)
            return beta_full_2loop(t, y)
        else:
            if self.loops == 1:
                return beta_nobtau_1loop(t, y)
            return beta_nobtau_2loop(t, y)

    def run(self, **kwargs):
        """
        Solve the RGEs over the t-range. kwargs passed to solve_ivp.
        """
        sol = solve_ivp(self.beta, self.t_span, self.y0, **kwargs)
        return sol

    def plot(self, sol=None):
        """
        Plot running couplings vs. scale.
        """
        if sol is None:
            sol = self.run()
        t_vals = sol.t
        mu_vals = defs.M_Z * np.exp(t_vals)

        # choose indexing depending on include_btau
        labels = ['g1', 'g2', 'g3', 'yt', 'yb', 'ytau', 'lambda'] if self.include_btau else ['g1', 'g2', 'g3', 'yt', 'lambda']
        y_vals = sol.y

        plt.figure()
        for i, lbl in enumerate(labels):
            plt.plot(mu_vals, y_vals[i], label=lbl)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'\mu [GeV]')
        plt.ylabel('Couplings')
        plt.legend()
        plt.tight_layout()
        plt.show()
