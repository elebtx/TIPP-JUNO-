import numpy as np
import matplotlib.pyplot as plt  

# 0) Constantes & config 

N_P = 1.45e33  # nombre de protons libres dans le détecteur (cibles IBD)

MEV_TO_J = 1.6e-13  # joules par MeV
E_FISSION_MEV = 200.0  # ce que libère la fission (~200 MeV)
E_FISSION_J = E_FISSION_MEV * MEV_TO_J  # énergie par fission en Joules

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY  # durée moyenne d'une année
LIVE_TIME_YEARS = 6.0  # temps d'exposition
T_EXP = LIVE_TIME_YEARS * SECONDS_PER_YEAR  # temps total en secondes

# Fenêtre énergie analysée (seuil IBD ~1.8 MeV)
E_MIN, E_MAX = 1.8, 8.0
N_BINS = 200  # nombre de bins pour l'histogramme final
E_BINS = np.linspace(E_MIN, E_MAX, N_BINS+1)  # bords des bins

# Résolution (sigma/E = a/sqrt(E))
def sigma_E_juno(E, a=0.03):
    E = np.clip(E, 1e-6, None)  # pr eviter division  par zero 
    return a * np.sqrt(E)       # sigma absolu en MeV

DM2_SCAN = np.linspace(2.36e-3, 2.48e-3, 61)  # valeurs testées
DM2_TRUE = 2.43e-3  # valeur "vraie" supposée


# 1) Spectres antineutrinos par fission

COEFFS = {
    "U235":  [3.217,  -3.111,  1.395,  -3.690e-1,  4.445e-2,  -2.053e-3],
    "U238":  [4.833e-1,  1.927e-1, -1.283e-1, -6.762e-3,  2.233e-3, -1.536e-4],
    "Pu239": [6.413,  -7.432,  3.535,  -8.820e-1,  1.025e-1, -4.550e-3],
    "Pu241": [3.251,  -3.204,  1.428,  -3.675e-1,  4.254e-2, -1.896e-3],
}  # coeff polynomiaux pr parametrer le spectre de chaque isotope

FISSION_FRACTIONS = {
    "U235": 0.577,
    "U238": 0.076,
    "Pu239": 0.295,
    "Pu241": 0.052
}  # poids de chaque isotope dans le spectre total

def antinu_spectrum_per_fission(E, isotope):
    # spectre ν̄_e par fission pour un isotope donné
    a = np.array(COEFFS[isotope], float)  # les 6 coeff
    powers = np.vstack([E**k for k in range(6)])  # E^0 à E^5
    return np.exp(np.dot(a, powers))  # 1/(MeV fission)

def total_flux_per_fission(E):
    # combine tt les isotopes en spectre tot par fission
    tot = np.zeros_like(E)
    for iso, frac in FISSION_FRACTIONS.items():
        tot += frac * antinu_spectrum_per_fission(E, iso)
    return tot  # 1/(MeV fission)


# 2) Section efficace
ME = 0.511
DELTA = 1.293  # en MeV

def sigma_ibd(E):
    # section efficace ν̄_e + p -> e+ + n (approximation simple)
    Ee = E - DELTA                      # énergie du positron
    pe2 = Ee**2 - ME**2                 # impulsion^2
    out = np.zeros_like(E)
    m = (Ee > ME) & (E >= 1.806)        # condition cinématique IBD
    out[m] = 1e-43 * np.sqrt(pe2[m]) * Ee[m]
    return out


# 3) Oscillations 
SIN2_TH12 = 0.307
SIN2_TH13 = 0.024
DM21 = 7.54e-5

def pee_article(E_nu, dm2ee, L_km, ordering="NO"):
    # probabilité de survie ν̄_e -> ν̄_e (formule type JUNO)

    c12_2 = 1.0 - SIN2_TH12
    c13_2 = 1.0 - SIN2_TH13

    sin2_2th12 = 4.0 * SIN2_TH12 * c12_2
    sin2_2th13 = 4.0 * SIN2_TH13 * c13_2

    E_GeV = E_nu / 1000.0  # convertion en GeV 

    # phases d'oscillation
    D21 = 1.267 * DM21 * L_km / E_GeV
    Dee = 1.267 * dm2ee * L_km / E_GeV

    A = 1.0 - sin2_2th12 * np.sin(D21)**2
    A = np.maximum(A, 1e-15)  # pr eviter A neg
    sqrtA = np.sqrt(A)

    # phase d'interférence sensible à la hiérarchie
    sinphi = (c12_2 * np.sin(2.0 * SIN2_TH12 * D21)
              - SIN2_TH12 * np.sin(2.0 * c12_2 * D21)) / sqrtA
    cosphi = (c12_2 * np.cos(2.0 * SIN2_TH12 * D21)
              + SIN2_TH12 * np.cos(2.0 * c12_2 * D21)) / sqrtA
    phi = np.arctan2(sinphi, cosphi)

    sign = +1.0 if ordering.upper() in ["NO", "NH", "NORMAL"] else -1.0

    term_sol = (c13_2**2) * sin2_2th12 * np.sin(D21)**2
    term_atm = 0.5 * sin2_2th13 * (1.0 - sqrtA * np.cos(2.0 * Dee + sign * phi))
    return np.clip(1.0 - term_sol - term_atm, 0.0, 1.0)


# 4) Efficiency

def efficiency(E, E_turn=2.2, width=0.15, plateau=0.73):
    # efficacité : montée progressive + plateau
    return plateau / (1.0 + np.exp(-(E - E_turn) / width))


# 5) Réacteurs: IDEAL vs REAL 
REACTORS_IDEAL = [{"P_GW": 36.0, "L_km": 52.5}]

REACTORS_REAL = [
    {"P_GW": 2.9, "L_km": 52.75},
    {"P_GW": 2.9, "L_km": 52.84},
    {"P_GW": 2.9, "L_km": 52.42},
    {"P_GW": 2.9, "L_km": 52.51},
    {"P_GW": 2.9, "L_km": 52.12},
    {"P_GW": 2.9, "L_km": 52.21},
    {"P_GW": 4.6, "L_km": 52.76},
    {"P_GW": 4.6, "L_km": 52.63},
    {"P_GW": 4.6, "L_km": 52.32},
    {"P_GW": 4.6, "L_km": 52.20},
]  # ici cas réalistes avec 10 coeurs 

def reactor_weights(reactors):
    # poids de chaque coeur en P/L**2
    P = np.array([r["P_GW"] for r in reactors], dtype=float)
    L = np.array([r["L_km"] for r in reactors], dtype=float)
    w = P / (L**2)
    w /= w.sum()
    return w


# 6) pas d'oscillation + cas ideal

def base_rate_density_noosc_ideal(E):
    # taux différentiel sans oscillation
    P_GW = 36.0
    L_km = 52.5
    L_cm = L_km * 1e5

    fission_rate = (P_GW * 1e9) / E_FISSION_J  # s^-1 nbr fission par sec
    geom = 1.0 / (4.0 * np.pi * L_cm**2)       # cm^-2
    coeff = fission_rate * geom                # flux incident

    spectrum = total_flux_per_fission(E)
    sig = sigma_ibd(E)
    eps = efficiency(E)

    return N_P * coeff * spectrum * sig * eps  # events/(MeV s)

def expected_total_events_noosc_ideal():
    # on integre sur l'nrj pr avoir le taux tot event/sec
    E = np.linspace(E_MIN, E_MAX, 12000)
    r = np.trapezoid(base_rate_density_noosc_ideal(E), E)
    return r * T_EXP  # pr 6 ans


# 7) monte carlo (UNE fois)

def sample_E_true_noosc(N, seed=0):
    rng = np.random.default_rng(seed)  # generateur aleatoire
    grid = np.linspace(E_MIN, E_MAX, 12000)
    f = base_rate_density_noosc_ideal(grid)
    fmax = f.max()  # cherche un majorante f max

    out = []
    while len(out) < N:
        batch = 300000
        E = rng.uniform(E_MIN, E_MAX, size=batch)  # propose des nrj uniforme
        u = rng.uniform(0.0, fmax, size=batch)
        fb = base_rate_density_noosc_ideal(E)
        keep = E[u < fb]  # condition d'acceptation
        out.extend(keep.tolist())

    return np.array(out[:N])

def make_master_sample(a_res=0.03, seed=0, scale_stats=0.6):
    # calcule le nbr d'evenement no osc sur 6 ans
    rng = np.random.default_rng(seed)
    mu0 = expected_total_events_noosc_ideal()
    N = int(np.round(scale_stats * mu0))

    E_true = sample_E_true_noosc(N, seed=seed)
    E_rec = rng.normal(loc=E_true, scale=sigma_E_juno(E_true, a=a_res))

    m = np.isfinite(E_rec) & (E_rec >= E_MIN) & (E_rec <= E_MAX)
    return E_true[m], E_rec[m], mu0


# 8) "ideal" vs "real"

def pee_eff(E_true, dm2ee, ordering, reactors):
    # proba de survie effective (moyenne sur les coeurs)
    w = reactor_weights(reactors)
    Ls = np.array([r["L_km"] for r in reactors], dtype=float)
    out = np.zeros_like(E_true, dtype=float)
    for wr, L in zip(w, Ls):
        out += wr * pee_article(E_true, dm2ee=dm2ee, L_km=L, ordering=ordering)
    return out

def template_from_master(E_true_m, E_rec_m, mu0_noosc, dm2ee, ordering, reactors):
    # construit un template en énergie reconstruite
    w = pee_eff(E_true_m, dm2ee=dm2ee, ordering=ordering, reactors=reactors)
    counts_w, _ = np.histogram(E_rec_m, bins=E_BINS, weights=w)
    scale = mu0_noosc / max(len(E_true_m), 1)
    return counts_w * scale


# 9) Chi2 
def chi2_ideal(M, T):
    # formule chi2 page43 (statistique seule)
    M = np.clip(M, 1e-12, None)
    return np.sum((M - T)**2 / M)

def chi2_real_fast(M, T, sigma_shape=0.01, sigma_alpha=0.022):
    # on inclu deux types d'incertitudes :
    # - sigma shape : incertitude de forme
    # - sigma alpha : incertitude de normalisation
    alphas = np.linspace(0.95, 1.05, 241)
    best = (np.inf, None)
    M = np.asarray(M, dtype=float)

    for a in alphas:
        mu = a * T
        var = np.clip(M + (sigma_shape * mu)**2, 1e-12, None)
        c2 = np.sum((M - mu)**2 / var) + ((a - 1.0) / sigma_alpha)**2
        if c2 < best[0]:
            best = (c2, a)

    return best


# 10) Plot

def make_plot(true_ordering="NO", dm2_true=DM2_TRUE, dm2_scan=DM2_SCAN, a_res=0.03, seed=0):
    # true_ordering : hiérarchie vraie utilisée pour l'Asimov

    if true_ordering.upper() in ["IO", "IH", "INVERTED"]:
        dm2_true = -abs(dm2_true)
        dm2_scan = -abs(dm2_scan)

    E_true_m, E_rec_m, mu0 = make_master_sample(a_res=a_res, seed=seed, scale_stats=0.6)

    # Asimov 
    M_ideal = template_from_master(E_true_m, E_rec_m, mu0, dm2_true,
                                   true_ordering, REACTORS_IDEAL).copy()
    M_real  = template_from_master(E_true_m, E_rec_m, mu0, dm2_true,
                                   true_ordering, REACTORS_REAL).copy()

    chi2_NO_i, chi2_IO_i = [], []
    chi2_NO_r, chi2_IO_r = [], []

    for dm2 in dm2_scan:
        T_NO_i = template_from_master(E_true_m, E_rec_m, mu0, dm2, "NO", REACTORS_IDEAL)
        T_IO_i = template_from_master(E_true_m, E_rec_m, mu0, dm2, "IO", REACTORS_IDEAL)

        T_NO_r = template_from_master(E_true_m, E_rec_m, mu0, dm2, "NO", REACTORS_REAL)
        T_IO_r = template_from_master(E_true_m, E_rec_m, mu0, dm2, "IO", REACTORS_REAL)

        chi2_NO_i.append(chi2_ideal(M_ideal, T_NO_i))
        chi2_IO_i.append(chi2_ideal(M_ideal, T_IO_i))
        chi2_NO_r.append(chi2_ideal(M_real, T_NO_r))
        chi2_IO_r.append(chi2_ideal(M_real, T_IO_r))

    chi2_NO_i = np.array(chi2_NO_i)
    chi2_IO_i = np.array(chi2_IO_i)
    chi2_NO_r = np.array(chi2_NO_r)
    chi2_IO_r = np.array(chi2_IO_r)

    if true_ordering.upper() in ["NO", "NH", "NORMAL"]:
        true_i, false_i = chi2_NO_i, chi2_IO_i
        true_r, false_r = chi2_NO_r, chi2_IO_r
        title = "NO"
    else:
        true_i, false_i = chi2_IO_i, chi2_NO_i
        true_r, false_r = chi2_IO_r, chi2_NO_r
        title = "IO"

    y_true_i  = true_i  - true_i.min()
    y_false_i = false_i - true_i.min()
    y_true_r  = true_r  - true_r.min()
    y_false_r = false_r - true_r.min()

    x = np.abs(dm2_scan) * 1e3

    plt.figure(figsize=(7.2, 5.2))
    plt.plot(x, y_true_i,  "k-",  lw=2, label="bonne hierarchie(ideal)")
    plt.plot(x, y_true_r,  "k--", lw=2, label="bonne hierarchie (real)")
    plt.plot(x, y_false_i, "r-",  lw=2, label="mauvaise hierarchie(ideal)")
    plt.plot(x, y_false_r, "r--", lw=2, label="mauvaise hierarchie(real)")

    plt.xlabel(r"$|\Delta m^2_{ee}| \ (\times 10^{-3}\ \mathrm{eV}^2)$")
    plt.ylabel(r"$\Delta\chi^2$")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.ylim(0, 25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    make_plot(true_ordering="NO", a_res=0.03, seed=1)

