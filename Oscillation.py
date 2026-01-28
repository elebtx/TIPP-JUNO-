import numpy as np

# Paramètres standards (PDG)
DEFAULT_PARAMS = {
    'theta12': np.arcsin(np.sqrt(0.307)),       # sin^2(t12) = 0.307
    'theta13': np.arcsin(np.sqrt(0.0218)),      # sin^2(t13) = 0.0218
    'dm2_21': 7.53e-5,                          # eV^2
    'dm2_31': 2.528e-3,                         # eV^2 (magnitude absolue)
}

def survival_probability(energy_MeV, baseline_km, epsilon):
    """
    Calcule la probabilité de survie P(nue -> nue) pour 3 saveurs.
    """
    params = DEFAULT_PARAMS

    # Récupération des paramètres
    t12 = params['theta12']
    t13 = params['theta13']
    dm2_21 = params['dm2_21']
    
    # Application de la hiérarchie de masse :
    # epsilon = 1  => dm2_31 > 0 (NO)
    # epsilon = -1 => dm2_31 < 0 (IO)
    dm2_31 = epsilon * abs(params['dm2_31'])
    
    dm2_32 = dm2_31 - dm2_21  # Relation de cohérence

    # Facteur de conversion des unités
    # 1.267 * dm2(eV^2) * L(km) / E(GeV)
    # Ici E est en MeV, donc facteur * 1000 => 1267
    const = 1267.0
    
    # On ajoute 1e-9 à E pour éviter la division par zéro
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
    
    prob = 1 - term_solar - term_atmos
    return prob

def oscillation_phase():
    """
    Fonction future qui calculerait la phase d'oscillation.
    """
    pass