import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from pathlib import Path

from matplotlib import rcParams as rc
# import gridspec
import matplotlib.gridspec as gridspec


rc["mathtext.fontset"] = "stix"
rc["font.family"] = "STIXGeneral"

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rc("text", usetex=True)


# If global_model_package is not installed as package with pip install -e . , adds the global_model_package to the path so that it can be imported as a package
try :
    import global_model_package
    print("'global_model_package' imported as pip package or already in sys.path.")
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.reactions import ElectronHeatingConstantRFPower

from config import config_dict
from reaction_set_N_et_O import get_species_and_reactions

final_states_per_power = {}
final_states_list = []

# Lists to store results for plotting
density_ne = []
density_O = []
density_Op = []
density_O2 = []
density_O2p = []
density_N = []
density_Np = []
density_N2 = []
density_N2p = []
# Thrust_ni = []
Thrust_ni_1 = []
Thrust_ng = []
Temp_e = []
Temp_ng_mono = []
Temp_ng_diato = []
Power_transfer_efficiency = []
Gamma_efficiency = []
xi_efficiency = []
eta_efficiency = []

power = 3000
# Solve the model
altitudes = np.arange(100, 281, 20)  # in km
for altitude in altitudes:
    chamber = Chamber(config_dict)
    species, initial_state, reactions_list, electron_heating = get_species_and_reactions(chamber, altitude)
    log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("logs")
    # model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="N_O_simple_thruster_constant_kappa", log_folder_path=log_folder_path)
    # comp_data = pd.read_csv("comp_atm_ready.txt", sep ="\t")
    comp_data = pd.read_csv("comp_atm_nrlmsise00_ready.txt", sep ="\t")
    print(comp_data.columns)
    comp_data = comp_data[comp_data["Heit(km)"] == altitude]

    electron_heating = ElectronHeatingConstantRFPower(species, power, chamber)
    model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="NO"+str(power)+"_alt_"+str(altitude), log_folder_path=log_folder_path)
    try:
        print("Solving model...")
        sol = model.solve(0, 1, initial_state)  # TODO Needs some testing
        print("Model resolved !")
        final_states_per_power[power] = sol.y[:, -1]
        print(final_states_per_power)
        print(final_states_list)
    except Exception as exception:
        print("Entering exception...")
        model.var_tracker.save_tracked_variables()
        print("Variables saved")
        raise exception

    # Then analyze final state
    final_states = sol.y
    density_ne.append(final_states[0][-1])
    density_O.append(final_states[7][-1])
    density_Op.append(final_states[8][-1])
    density_O2.append(final_states[6][-1])
    density_O2p.append(final_states[5][-1])
    density_N.append(final_states[2][-1])
    density_N2.append(final_states[1][-1])
    density_Np.append(final_states[4][-1])
    density_N2p.append(final_states[3][-1])
    Temp_e.append(final_states[species.nb][-1])
    Temp_ng_mono.append(final_states[species.nb + 1][-1])
    Temp_ng_diato.append(final_states[species.nb + 2][-1])
    h_L = chamber.h_L(final_states[1][-1])
    # Thrust_ni.append(h_L * density_ne[-1] * (e * Temp_e[-1])**.5 * (2 * e * chamber.V_grid)**.5 * chamber.R**2. * pi * chamber.beta_i)
    Thrust_ni_1.append(model.total_ion_thrust(final_states[:, -1]))
    Thrust_ng.append(model.total_neutral_thrust(final_states[:, -1]))
    Power_transfer_efficiency.append(electron_heating.power_transfer_efficiency)
    xi_efficiency.append((Thrust_ni_1[-1] + Thrust_ng[-1]) / power)

    for index in range(len(model.species.species[:])):
        print(model.species.species[index].name, final_states[index][-1])
    # calculate eta_efficiency
    nN2plus = final_states[3][-1]
    nNplus = final_states[4][-1]
    nO2plus = final_states[5][-1]
    nOplus = final_states[8][-1]

    h_L = chamber.h_L(final_states[1][-1] + final_states[2][-1] + final_states[6][-1]+ final_states[7][-1])
    print("h_L for eta :", h_L)

    MN2 = 4.65e-26  # mass of N2 ion in kg
    MN = 2.32e-26  # mass of N ion in kg
    MO2 = 5.32e-26  # mass of O2 ion in kg
    MO = 2.66e-26  # mass of O ion in kg
    u_BN2 = (e * Temp_e[-1] / MN2)**.5
    u_BN = (e * Temp_e[-1] / MN)**.5
    u_BO2 = (e * Temp_e[-1] / MO2)**.5
    u_BO = (e * Temp_e[-1] / MO)**.5

    collection_rate = 0.5
    injection_rates = collection_rate * np.array([2e12, comp_data["Q_N2(s-1)"].values[0], comp_data["Q_N(s-1)"].values[0], 1e12, 1e12, 0.0, comp_data["Q_O2(s-1)"].values[0], comp_data["Q_O(s-1)"].values[0], 0.0])
    print("injection rates:", injection_rates)

    AREA = pi * chamber.R**2. * 0.7

    eta_N2 = h_L * nN2plus * u_BN2 * AREA / injection_rates[1]
    eta_N = h_L * nNplus * u_BN * AREA / injection_rates[2]
    eta_O2 = h_L * nO2plus * u_BO2 * AREA / injection_rates[6]
    eta_O = h_L * nOplus * u_BO * AREA / injection_rates[7]

    eta_tot = (h_L * nN2plus * u_BN2 * AREA + h_L * nNplus * u_BN * AREA + h_L * nO2plus * u_BO2 * AREA + h_L * nOplus * u_BO * AREA) / (injection_rates[1] + injection_rates[2] + injection_rates[6] + injection_rates[7])


    gamma_i_N2plus = h_L * nN2plus * u_BN2 * AREA * e * chamber.V_grid / power
    gamma_i_Nplus = h_L * nNplus * u_BN * AREA * e * chamber.V_grid / power
    gamma_i_O2plus = h_L * nO2plus * u_BO2 * AREA * e * chamber.V_grid / power
    gamma_i_Oplus = h_L * nOplus * u_BO * AREA * e * chamber.V_grid / power

    gamma_tot = gamma_i_N2plus + gamma_i_Nplus + gamma_i_O2plus + gamma_i_Oplus

    Gamma_efficiency.append(gamma_tot)

    print("eta N2plus :", eta_N2, h_L * nN2plus * u_BN2 * AREA, injection_rates[1])
    print("eta Nplus :", eta_N, h_L * nNplus * u_BN * AREA, injection_rates[2])
    print("eta O2plus :", eta_O2, h_L * nO2plus * u_BO2 * AREA, injection_rates[6])
    print("eta Oplus :", eta_O, h_L * nOplus * u_BO * AREA, injection_rates[7])
    print("eta total :", eta_tot)
    eta_efficiency.append(eta_tot)
    # eta_efficiency.append(gamma_i * chamber.R**2. * pi * chamber.beta_i / 


# create a pd database
df_results = pd.DataFrame({
    "Altitude (km)": altitudes,
    "n_e (m-3)": density_ne,
    "n_O (m-3)": density_O,
    "n_O2 (m-3)": density_O2,
    "n_O+ (m-3)": density_Op,
    "n_O2+ (m-3)": density_O2p,
    "n_N (m-3)": density_N,
    "n_N2 (m-3)": density_N2,
    "n_N+ (m-3)": density_Np,
    "n_N2+ (m-3)": density_N2p,
    "T_e (eV)": Temp_e,
    "T_ng_mono (eV)": Temp_ng_mono,
    "T_ng_diato (eV)": Temp_ng_diato,
    "Thrust_ion (N)": Thrust_ni_1,
    "Thrust_neutral (N)": Thrust_ng,
    "Power_transfer_efficiency": Power_transfer_efficiency,
    "Gamma_efficiency": Gamma_efficiency,
    "eta_efficiency": eta_efficiency,
    "xi_efficiency": xi_efficiency
})

df_results.to_csv(f"data/results_scan_altitude_{power}.csv", index=False)