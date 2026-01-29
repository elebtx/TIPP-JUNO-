import numpy as np

class PhysicsConstants:
    # Constantes Nucléaires et Physiques
    GW_TO_MEV_S = 1e9 * 6.241509e12 
    KM_TO_CM = 1e5
    DELTA_NP = 1.2933
    ME = 0.511
    SIGMA_PREFACTOR = 0.0952 # Correspond à des unités de 1e-42 cm^2
    
    # Constantes du Détecteur JUNO
    MASS_DETECTOR_KG = 20e6     # 20 kton
    H_MASS_FRACTION = 0.12      # ~12% d'hydrogène en masse
    NA = 6.022e23               # Avogadro
    MOLAR_MASS_H = 1.00794e-3   # kg/mol
    
    # Facteur de normalisation global : N_p * (Conversion Sigma) * (Sec/Jour)
    # N_p = (Masse * Frac / M_H) * NA
    N_PROTONS = (MASS_DETECTOR_KG * H_MASS_FRACTION / MOLAR_MASS_H) * NA
    SIGMA_UNIT_CORRECTION = 1e-42
    SECONDS_PER_DAY = 86400

ISOTOPES = {
    'U235':  {'e_fis': 202.36, 'frac': 0.58, 'coeffs': [0.870, -0.160, -0.0910]},
    'U238':  {'e_fis': 205.99, 'frac': 0.07, 'coeffs': [0.976, -0.162, -0.0790]},
    'Pu239': {'e_fis': 211.12, 'frac': 0.30, 'coeffs': [0.896, -0.239, -0.0981]},
    'Pu241': {'e_fis': 214.26, 'frac': 0.05, 'coeffs': [0.793, -0.080, -0.1085]}
}

class Reactor:
    def __init__(self, name, power_gw, baseline_km):
        self.name = name
        self.power_gw = power_gw
        self.baseline_km = baseline_km
        self.fission_fractions = {iso: data['frac'] for iso, data in ISOTOPES.items()}

    def get_fission_rate(self):
        denom = sum(self.fission_fractions[iso] * ISOTOPES[iso]['e_fis'] for iso in ISOTOPES)
        return (self.power_gw * PhysicsConstants.GW_TO_MEV_S) / denom


def spectrum_per_fission(energy, isotope):
    c = ISOTOPES[isotope]['coeffs']
    return np.exp(c[0] + c[1]*energy + c[2]*(energy**2))


# Calcul de la section efficace IBD
def ibd_cross_section(energy):
    threshold = PhysicsConstants.DELTA_NP + PhysicsConstants.ME
    sigma = np.zeros_like(energy)
    mask = energy > threshold
    
    E_e = energy[mask] - PhysicsConstants.DELTA_NP
    p_e = np.sqrt(E_e**2 - PhysicsConstants.ME**2)
    
    sigma[mask] = PhysicsConstants.SIGMA_PREFACTOR * E_e * p_e
    return sigma

# Calcul du nombre d'événements par jour
def calculate_events_per_day(energies, spectrum_y, efficiency=0.73):
    integral = np.trapezoid(spectrum_y, energies)
    
    # Conversion en taux d'événements
    # Taux = Intégrale * (Correction Unité Sigma) * N_Protons * (Sec/Jour) * Efficacité
    factor = (PhysicsConstants.SIGMA_UNIT_CORRECTION * PhysicsConstants.N_PROTONS * PhysicsConstants.SECONDS_PER_DAY * efficiency)
              
    return integral * factor
