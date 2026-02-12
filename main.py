import numpy as np
import matplotlib.pyplot as plt

import JunoPhysics as phys   
import Oscillation as osc   
import NeutrinoFlux as flux  


# Grille d'énergie
energies = np.linspace(1.8, 10, 500) 
dE = energies[1] - energies[0]
sigma = phys.ibd_cross_section(energies)

# Liste des 10 cœurs de réacteurs (Yangjiang + Taishan)
# Configuration nominale JUNO pour atteindre ~35.8 GWth
reactors = [
    phys.Reactor("YJ-C1", 2.9, 52.75), phys.Reactor("YJ-C2", 2.9, 52.84),
    phys.Reactor("YJ-C3", 2.9, 52.42), phys.Reactor("YJ-C4", 2.9, 52.51),
    phys.Reactor("YJ-C5", 2.9, 52.12), phys.Reactor("YJ-C6", 2.9, 52.21),
    phys.Reactor("TS-C1", 4.6, 52.76), phys.Reactor("TS-C2", 4.6, 52.63),
    phys.Reactor("TS-C3", 4.6, 52.32), phys.Reactor("TS-C4", 4.6, 52.20),
]

# Initialisation des flux
flux_total_no_osc = np.zeros_like(energies)
flux_total_with_osc_NO = np.zeros_like(energies)
flux_total_with_osc_IO = np.zeros_like(energies)

print(f"Simulation JUNO démarrée : {phys.PhysicsConstants.LIVE_TIME_YEARS} ans d'acquisition...")

# Boucle sur les réacteurs
for core in reactors:
    rate = core.get_fission_rate()
    dist_cm = core.baseline_km * phys.PhysicsConstants.KM_TO_CM
    geom = 1.0 / (4 * np.pi * dist_cm**2)
    
    # Spectre émis par le cœur
    core_spectrum = np.zeros_like(energies)
    for iso, frac in core.fission_fractions.items():
        core_spectrum += frac * phys.spectrum_per_fission(energies, iso)
    
    flux_pure = rate * geom * core_spectrum
    
    # Probabilités de survie (Normal et Inverted Ordering)
    prob_survie_NO = osc.survival_probability(energies, core.baseline_km, epsilon=1)
    prob_survie_IO = osc.survival_probability(energies, core.baseline_km, epsilon=-1)
    
    flux_total_no_osc += flux_pure
    flux_total_with_osc_NO += flux_pure * prob_survie_NO
    flux_total_with_osc_IO += flux_pure * prob_survie_IO

# MISE À L'ÉCHELLE DES SPECTRES (Events per bin)
norm_factor = (phys.PhysicsConstants.SIGMA_UNIT_CORRECTION * phys.PhysicsConstants.N_PROTONS * phys.PhysicsConstants.SECONDS_PER_DAY * phys.PhysicsConstants.EFFICIENCY)

# Facteur final : Taux/MeV/jour -> Evénements/bin/6ans
scaling = norm_factor * dE * (phys.PhysicsConstants.LIVE_TIME_YEARS * phys.PhysicsConstants.DAYS_PER_YEAR)

spec_no_osc = flux_total_no_osc * sigma * scaling
spec_oscNO = flux_total_with_osc_NO * sigma * scaling
spec_oscIO = flux_total_with_osc_IO * sigma * scaling

# Application de la résolution en énergie (3% / sqrt(E))
spec_oscNO_res = phys.energy_resolution(energies, spec_oscNO)
spec_oscIO_res = phys.energy_resolution(energies, spec_oscIO)


# Graphiques : Spectres théoriques et avec résolution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Graphique 1 : spectres théoriques 
ax1.plot(energies, spec_no_osc, 'gray', label="Sans Oscillation", alpha=0.4, linestyle='--')
ax1.plot(energies, spec_oscNO, 'red', linewidth=1.5, label="NO (Théorique)")
ax1.plot(energies, spec_oscIO, 'blue', linewidth=1.5, label="IO (Théorique)")

ax1.set_title("Spectres Théoriques (Sans résolution)", fontsize=13)
ax1.set_xlabel("Energie Neutrino (MeV)", fontsize=11)
ax1.set_ylabel(f"Events per bin (ΔE = {dE:.3f} MeV) over {phys.PhysicsConstants.LIVE_TIME_YEARS:g} years", fontsize=11)
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.2)
ax1.set_xlim(1.8, 8.5)
ax1.set_ylim(bottom=0)

# Graphique 2 : spectres réels (avec résolution) 
ax2.plot(energies, spec_no_osc, 'gray', label="Sans Oscillation", alpha=0.4, linestyle='--')
ax2.plot(energies, spec_oscNO_res, 'red', linewidth=1.8, label="Normal Ordering (NO)")
ax2.plot(energies, spec_oscIO_res, 'blue', linewidth=1.8, label="Inverted Ordering (IO)")

ax2.set_title(f"Spectres Réels (Résolution 3%/√E)", fontsize=13)
ax2.set_xlabel("Energie Neutrino (MeV)", fontsize=11)
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.2)
ax2.set_xlim(1.8, 8.5)

plt.suptitle(f"Acquisition JUNO : {phys.PhysicsConstants.LIVE_TIME_YEARS} ans", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Affichage du nombre total d'événements pour vérification
total_evts = np.sum(spec_oscNO_res)
print(f"Nombre total d'événements (NO) sur {phys.PhysicsConstants.LIVE_TIME_YEARS} ans : {total_evts:.0f}")
print(f"Soit environ {total_evts/(phys.PhysicsConstants.LIVE_TIME_YEARS*phys.PhysicsConstants.DAYS_PER_YEAR):.2f} événements / jour.")
