import numpy as np
import matplotlib.pyplot as plt

class PhysicsConstants:
    GW_TO_MEV_S = 1e9 * 6.241509e12  # Conversion Puissance -> MeV/s
    KM_TO_CM = 1e5                   # Conversion Distance -> cm
    
    # Constantes pour la section efficace IBD (Vogel & Beacom)
    DELTA_NP = 1.2933  # Différence de masse Neutron-Proton (MeV)
    ME = 0.511         # Masse de l'électron (MeV)
    SIGMA_PREFACTOR = 0.0952 # Préfacteur en 10^-42 cm^2/MeV^2

# Paramètres des isotopes
ISOTOPES = {
    'U235':  {'energy_per_fission': 202.36, 'fission_fraction': 0.58},
    'U238':  {'energy_per_fission': 205.99, 'fission_fraction': 0.07},
    'Pu239': {'energy_per_fission': 211.12, 'fission_fraction': 0.30},
    'Pu241': {'energy_per_fission': 214.26, 'fission_fraction': 0.05}
}

# Paramètres de Vogel-Engel pour le spectre S(E)
VOGEL_COEFFS = {
    'U235':  [0.870, -0.160, -0.0910],
    'U238':  [0.976, -0.162, -0.0790],
    'Pu239': [0.896, -0.239, -0.0981],
    'Pu241': [0.793, -0.080, -0.1085]
}

def ibd_cross_section(energy_nu):
    """
    Calcule la section efficace Inverse Beta Decay (IBD).
    """
    # Seuil de réaction : E_nu > Delta + m_e approx 1.806 MeV
    threshold = PhysicsConstants.DELTA_NP + PhysicsConstants.ME
    
    # Initialisation à zéro
    sigma = np.zeros_like(energy_nu)
    
    # Calcul uniquement pour les énergies au-dessus du seuil
    mask = energy_nu > threshold
    E_nu_valid = energy_nu[mask]
    
    # Energie du positron: E_e = E_nu - (Mn - Mp)
    E_e = E_nu_valid - PhysicsConstants.DELTA_NP
    
    # Impulsion du positron: p_e = sqrt(E_e^2 - m_e^2)
    p_e = np.sqrt(E_e**2 - PhysicsConstants.ME**2)
    
    # Formule : sigma = 0.0952 * E_e * p_e * 10^-42 cm^2
    val = PhysicsConstants.SIGMA_PREFACTOR * E_e * p_e
    
    sigma[mask] = val
    return sigma

def spectrum_per_fission(energy_array, isotope):
    """Spectre S_i(E) selon Vogel-Engel."""
    coeffs = VOGEL_COEFFS[isotope]
    return np.exp(coeffs[0] + coeffs[1]*energy_array + coeffs[2]*(energy_array**2))

class Reactor:
    def __init__(self, name, power_gw, baseline_km):
        self.name = name
        self.power_gw = power_gw
        self.baseline_km = baseline_km
        self.fission_fractions = {iso: data['fission_fraction'] for iso, data in ISOTOPES.items()}

    def get_fission_rate(self):
        denom = 0
        for iso, fraction in self.fission_fractions.items():
            e_i = ISOTOPES[iso]['energy_per_fission']
            denom += fraction * e_i
        return (self.power_gw * PhysicsConstants.GW_TO_MEV_S) / denom

def calculate_flux_components(reactors, energy_grid):
    """
    Retourne un dictionnaire avec le flux total et les contributions par isotope.
    """
    # Initialisation 
    fluxes = {iso: np.zeros_like(energy_grid) for iso in ISOTOPES.keys()}
    fluxes['Total'] = np.zeros_like(energy_grid)
    
    print(f"Calcul des fluxs pour {len(reactors)} coeurs...")
    
    for core in reactors:
        fission_rate = core.get_fission_rate()
        dist_cm = core.baseline_km * PhysicsConstants.KM_TO_CM
        geom_factor = 1.0 / (4 * np.pi * dist_cm**2)
        
        # Pour chaque isotope, on ajoute sa contribution
        for iso, fraction in core.fission_fractions.items():
            s_i = spectrum_per_fission(energy_grid, iso)
            
            # Flux partiel = Taux * Fraction * Spectre * Géométrie
            partial_flux = fission_rate * fraction * s_i * geom_factor
            
            fluxes[iso] += partial_flux
            fluxes['Total'] += partial_flux
            
    return fluxes

def observable_spectrum(fluxes_dict, sigma_array):
    """
    Calcule le spectre observable (Taux d'événements) pour chaque isotope.
    Effectue le produit : Flux(E) * SectionEfficace(E).
    """
    obs_spectra = {}
    
    for isotope, flux_array in fluxes_dict.items():
        obs_spectra[isotope] = flux_array * sigma_array
        
    return obs_spectra

# Liste des réacteurs (cf tableau 1-2)
reactor_list = [
    Reactor("YJ-C1", 2.9, 52.75), Reactor("YJ-C2", 2.9, 52.84),
    Reactor("YJ-C3", 2.9, 52.42), Reactor("YJ-C4", 2.9, 52.51),
    Reactor("YJ-C5", 2.9, 52.12), Reactor("YJ-C6", 2.9, 52.21),
    Reactor("TS-C1", 4.6, 52.76), Reactor("TS-C2", 4.6, 52.63),
    Reactor("DYB", 17.4, 215),    Reactor("HZ", 17.4, 265),
]

energies = np.linspace(1.8, 10, 500)
flux_data = calculate_flux_components(reactor_list, energies) 
sigma_ibd = ibd_cross_section(energies)                       
event_rates = observable_spectrum(flux_data, sigma_ibd)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4))

styles = {
    'Total': {'c': 'black', 'ls': '-', 'lw': 2, 'label': 'Total'},
    'U235':  {'c': 'blue',  'ls': '--', 'lw': 1.5, 'label': 'U-235'},
    'U238':  {'c': 'magenta', 'ls': ':', 'lw': 1.5, 'label': 'U-238'},
    'Pu239': {'c': 'red',   'ls': '--', 'lw': 1.5, 'label': 'Pu-239'},
    'Pu241': {'c': 'green', 'ls': ':', 'lw': 1.5, 'label': 'Pu-241'}
}

for name, flux in flux_data.items():
    s = styles.get(name, {'c': 'gray', 'ls': '-'})
    ax_left.plot(energies, flux, label=s['label'], color=s['c'], linestyle=s['ls'], linewidth=s['lw'])

ax_left.set_xlabel("Energie neutrino (MeV)")
ax_left.set_ylabel(r"Number of $\bar{\nu}_e$ per cm$^2$ per s per MeV")
ax_left.set_ylim(bottom=0)
ax_left.legend(loc='upper center')
ax_left.grid(True, alpha=0.8)

# Axe de droite : Section Efficace
ax_left2 = ax_left.twinx()
ax_left2.plot(energies, sigma_ibd, color='gray', label=r'$\sigma_{IBD}$ (Vogel & Beacom)')
ax_left2.set_ylabel(r"Section Efficace IBD ($10^{-42} cm^2$)", color='black')
ax_left2.tick_params(axis='y', labelcolor='black')
ax_left2.text(8.5, sigma_ibd[-1], r'$\sigma_{IBD}$', color='gray', fontweight='bold')

for name, rate in event_rates.items():
    s = styles.get(name, {'c': 'gray', 'ls': '-'})
    ax_right.plot(energies, rate, label=s['label'], color=s['c'], linestyle=s['ls'], linewidth=s['lw'])

ax_right.set_xlabel("Energie neutrino (MeV)")
ax_right.set_ylabel(r"Taux d'événements IBD (événements par cm$^2$ par s par MeV)")
ax_right.set_ylim(bottom=0)
ax_right.legend(loc='upper right')
ax_right.grid(True, alpha=0.8)

plt.tight_layout()
plt.show()