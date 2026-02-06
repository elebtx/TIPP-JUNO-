import numpy as np

# Paramètres pour la hiérarchie normale (Normal Ordering) 
PARAMS_NO = {
    'theta12': np.arcsin(np.sqrt(0.308)),       # sin^2(t12) = 3.08e-1
    'theta13': np.arcsin(np.sqrt(0.0234)),      # sin^2(t13) = 2.34e-2
    'dm2_21': 7.54e-5,                          # Delta m^2_21 = 7.54e-5 eV^2
    'dm2_31': 2.47e-3,                          # Delta m^2_31 = 2.47e-3 eV^2
}

# Paramètres pour la hiérarchie inversée (Inverted Ordering) 
PARAMS_IO = {
    'theta12': np.arcsin(np.sqrt(0.308)),       # sin^2(t12) = 3.08e-1
    'theta13': np.arcsin(np.sqrt(0.0240)),      # sin^2(t13) = 2.40e-2
    'dm2_21': 7.54e-5,                          # Delta m^2_21 = 7.54e-5 eV^2
    'dm2_31': -2.42e-3,                         # Delta m^2_13 = 2.42e-3 eV^2 (donnée en valeur absolue dans le tableau)
}

def calculate_probability(energy_MeV, baseline_km, params):
    """
    Fonction générique de calcul de probabilité de survie.
    """
    t12 = params['theta12']
    t13 = params['theta13']
    dm2_21 = params['dm2_21']
    dm2_31 = params['dm2_31']
    dm2_32 = dm2_31 - dm2_21 

    const = 1267.0
    inv_E = 1.0 / (energy_MeV + 1e-9)
    
    delta_21 = const * dm2_21 * baseline_km * inv_E
    delta_31 = const * dm2_31 * baseline_km * inv_E
    delta_32 = const * dm2_32 * baseline_km * inv_E
    
    # Terme solaire
    term_solar = (np.cos(t13)**4) * (np.sin(2*t12)**2) * (np.sin(delta_21)**2)
    
    # Terme atmosphérique 
    term_atmos = (np.sin(2*t13)**2) * (
        (np.cos(t12)**2) * (np.sin(delta_31)**2) + 
        (np.sin(t12)**2) * (np.sin(delta_32)**2)
    )
    
    return 1 - term_solar - term_atmos


def survival_probability(energy_MeV, baseline_km, epsilon):
    if epsilon >= 0:
        return calculate_probability(energy_MeV, baseline_km, PARAMS_NO)
    else:
        return calculate_probability(energy_MeV, baseline_km, PARAMS_IO)
