import numpy as np
import matplotlib.pyplot as plt

import JunoPhysics as phys   
import Oscillation as osc     

energies = np.linspace(1.8, 10, 10000) 
sigma = phys.ibd_cross_section(energies)

reactors = [
    phys.Reactor("YJ-C1", 2.9, 52.75), phys.Reactor("YJ-C2", 2.9, 52.84),
    phys.Reactor("YJ-C3", 2.9, 52.42), phys.Reactor("YJ-C4", 2.9, 52.51),
    phys.Reactor("YJ-C5", 2.9, 52.12), phys.Reactor("YJ-C6", 2.9, 52.21),
    phys.Reactor("TS-C1", 4.6, 52.76), phys.Reactor("TS-C2", 4.6, 52.63),
    phys.Reactor("DYB", 17.4, 215),    phys.Reactor("HZ", 17.4, 265),
]

flux_total_no_osc = np.zeros_like(energies)
flux_total_with_osc_NO = np.zeros_like(energies)
flux_total_with_osc_IO = np.zeros_like(energies)
total_events_no_osc = 0
total_events_osc_NO = 0

print("Démarrage de la simulation JUNO...")

for core in reactors:
    rate = core.get_fission_rate()
    dist_cm = core.baseline_km * phys.PhysicsConstants.KM_TO_CM
    geom = 1.0 / (4 * np.pi * dist_cm**2)
    
    # Spectre émis par un coeur 
    core_spectrum = np.zeros_like(energies)
    for iso, frac in core.fission_fractions.items():
        core_spectrum += frac * phys.spectrum_per_fission(energies, iso)
    
    flux_pure = rate * geom * core_spectrum
    
    prob_survie_NO = osc.survival_probability(energies, core.baseline_km, epsilon=1)
    prob_survie_IO = osc.survival_probability(energies, core.baseline_km, epsilon=-1)
    
    flux_total_no_osc += flux_pure
    flux_total_with_osc_NO += flux_pure * prob_survie_NO
    flux_total_with_osc_IO += flux_pure * prob_survie_IO

    # Calcul des spectres par réacteur
    spec_pure_core = flux_pure * sigma
    spec_NO_core = flux_pure * prob_survie_NO * sigma
    
    # Somme des événements par jour pour ce réacteur
    total_events_no_osc += phys.calculate_events_per_day(energies, spec_pure_core)
    total_events_osc_NO += phys.calculate_events_per_day(energies, spec_NO_core)


spec_no_osc = flux_total_no_osc * sigma
spec_oscNO = flux_total_with_osc_NO * sigma
spec_oscIO = flux_total_with_osc_IO * sigma 

print(f"Nombre d'événements détecté par jour  : {total_events_osc_NO:.2f}")

# Application de la résolution
spec_oscNO_res = phys.energy_resolution(energies, spec_oscNO)
spec_oscIO_res = phys.energy_resolution(energies, spec_oscIO)

# Tracé du spectre après oscillation avec résolution
plt.figure(figsize=(10, 6))

#plt.plot(energies, spec_no_osc, 'black', label="Sans Oscillation", alpha=0.5)
plt.plot(energies, spec_oscNO, 'blue', linewidth=1, label="NO sans Résolution")
plt.plot(energies, spec_oscIO, 'red', linewidth=1, label="IO sans Résolution")
plt.plot(energies, spec_oscNO_res, '--', label="NO avec Résolution (3%)", alpha=0.8)
plt.plot(energies, spec_oscIO_res, '--', label="IO avec Résolution (3%)", alpha=0.8)


plt.title("Spectre JUNO", fontsize=14)
plt.xlabel("Energie (MeV)")
plt.ylabel("Evénements attendus (u.a.)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


