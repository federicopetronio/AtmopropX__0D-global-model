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


altitude = 250
chamber = Chamber(config_dict)
species, initial_state, reactions_list, electron_heating = get_species_and_reactions(chamber, altitude)
log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("logs")

# Solve the model
power = 1500
electron_heating = ElectronHeatingConstantRFPower(species, power, chamber)
model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="NO"+str(power)+"_alt_"+str(altitude), log_folder_path=log_folder_path)
try:
    print("Solving model...")
    sol = model.solve(0, 1, initial_state)  # TODO Needs some testing
    print("Model resolved !")
except Exception as exception:
    print("Entering exception...")
    model.var_tracker.save_tracked_variables()
    print("Variables saved")
    raise exception

# Then analyze final state
final_states = sol.y

final_states = sol.y
time_points = sol.t

columns = species.names + ['Te', 'Tmono', 'Tdiato']

df_temporal = pd.DataFrame(final_states.T, columns=columns)
df_temporal['Time (s)'] = time_points
df_temporal.to_csv(f"data/temporal_evolution_NO_{power}W_alt_{altitude}km.csv", index=False)
print("saved to: ", f"data/temporal_evolution_NO_{power}W_alt_{altitude}km.csv")


# # print(",".join(map(str, sol.y[:, -1])))
# final_state = sol.y[:, -1]
# #avg_e_density = 
# # print("h_L : ", chamber.h_L(final_state[1:].sum()))
# # print("h_R : ", chamber.h_R(final_state[1:].sum()))

# # Extract time points

# # Create figure and primary axis
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# # Plot species concentrations on the first subplot
# #ax1.yscale('log')
# for i, specie in enumerate(species.species):
#     ax1.loglog(time_points, final_states[i], label=specie.name)
# ax1.set_ylabel(r'Density of species m$^{-3}$)')
# ax1.legend(loc='best')
# ax1.grid()
# #ax1.grid(which='both', axis='y')

# # Create a secondary y-axis for Xenon temperature
# ax3 = ax2.twinx()

# # Plot temperatures: Electron on primary y-axis, Xenon on secondary y-axis
# ax2.loglog(time_points, final_states[species.nb], label='Electron Temp (eV)', color='blue')
# for i in range(1,3):
#     ax3.loglog(time_points, final_states[species.nb + i], linestyle='--', label= f"Molecules with {i} atoms Temp (eV)")

# ax2.set_ylabel('Electron Temperature', color='blue')
# ax3.set_ylabel('Molecules Temperature', color='red')
# ax2.tick_params(axis='y', labelcolor='blue')
# ax3.tick_params(axis='y', labelcolor='red')

# # Combine legends
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# lines_3, labels_3 = ax3.get_legend_handles_labels()
# ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc='best')

# # Set labels and title
# ax2.set_xlabel('Time (s)')
# ax1.set_title('Species Concentrations Over Time')
# ax2.set_title('Temperature Evolution')
# ax2.grid()

# # Show the plot
# plt.tight_layout()
# # plt.show()

# plt.figure()
# # plt.semilogy(power_list, np.array(Thrust_ni)*1.e3, label='Ion Thrust Estimate')
# plt.semilogy(power_list, np.array(Thrust_ni_1)*1.e3, label='Ion Thrust', ls=":")
# plt.semilogy(power_list, np.array(Thrust_ng)*1.e3, label='Neutral Thrust', ls="--")
# plt.xlabel('RF Power [W]')
# plt.ylabel('Thrust [$10^{-3}$ N]')
# plt.title('Ion Thrust Estimate vs RF Power')
# plt.legend()
# plt.grid()
# plt.ylim(1e-2, 10)
# plt.xlim(500, 3000)
# plt.tight_layout()
# plt.savefig(f"images/Thrust_vs_power.png")

# plt.figure()
# plt.plot(power_list, Gamma_efficiency, label='Power Efficiency')
# plt.plot(power_list, eta_efficiency, label='Mass efficiency', ls="--")
# plt.plot(power_list, Power_transfer_efficiency, label='Power transfer efficiency', ls=":")
# plt.ylim(0, 1)
# plt.xlim(500, 3000)
# plt.xlabel('RF Power [W]')
# # plt.ylabel(r'Power Transfer Efficiency $\zeta$')
# plt.ylabel('Power Transfer Efficiency')
# plt.title('Power Transfer Efficiency vs RF Power')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"images/Power_efficiency_vs_power.png")

# plt.figure()
# ax1 = plt.gca()
# ax2 = ax1.twinx()
# ax1.plot(power_list, Temp_e, label='Electron Temperature', marker="o", color='blue')
# ax2.plot(power_list, Temp_ng_mono, label='Monoatomic Neutral Temperature', marker="s", color='red')
# ax2.plot(power_list, Temp_ng_diato, label='Diatomic Neutral Temperature', marker="^", color='green')
# ax1.set_xlabel('RF Power [W]')
# ax1.set_ylabel('Electron Temperature [eV]')
# ax2.set_ylabel('Neutral Temperature [eV]')
# ax1.set_title('Temperatures vs RF Power')
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
# ax1.set_xlim(500, 3000)
# ax1.set_ylim(0, 30)
# ax2.set_ylim(0, .05)
# ax1.grid()
# plt.tight_layout()
# plt.savefig(f"images/Temperatures_vs_power.png")


# # plot density_ne and density_ng vs power_list

# plt.figure(figsize=(7,7))
# plt.subplot(211)
# plt.semilogy(power_list, density_O, label='O Density', marker="s")
# plt.semilogy(power_list, density_O2, label=r'O$_2$ Density', marker="^")
# plt.semilogy(power_list, density_N, label='N Density', marker="d")
# plt.semilogy(power_list, density_N2, label=r'N$_2$ Density', marker="v")
# plt.xlabel('RF Power [W]')
# plt.ylabel(r'Density [m$^{-3}$]')
# plt.title('Densities vs RF Power')
# plt.grid()

# plt.legend()

# plt.subplot(212)
# plt.semilogy(power_list, density_ne, label='Electron Density', marker="o")
# plt.semilogy(power_list, density_Op, label=r'O$^+$ Density', marker="*")
# plt.semilogy(power_list, density_O2p, label=r'O$_2^+$ Density', marker="x")
# plt.semilogy(power_list, density_Np, label=r'N$^+$ Density', marker="h")
# plt.semilogy(power_list, density_N2p, label=r'N$_2^+$ Density', marker="D")
# plt.semilogy(np.array(power_list), np.array(density_Op) + np.array(density_Np) + np.array(density_N2p) + np.array(density_O2p), label='Total Ion Density', marker="P")
# plt.xlabel('RF Power [W]')
# plt.ylabel(r'Density [m$^{-3}$]')
# plt.title('Electron Density vs RF Power')
# plt.legend()

# plt.grid()
# plt.tight_layout()
# plt.savefig(f"images/Densities_vs_power.png")


# # Plot time evolution of species and temperatures
# plt.figure(figsize=(7,5))
# plt.plot(time_points, final_states[0], label='e-', marker="o")
# plt.plot(time_points, final_states[1], label='N2', marker="s")
# plt.plot(time_points, final_states[2], label='N', marker="^")
# plt.plot(time_points, final_states[6], label='O2', marker="s")
# plt.plot(time_points, final_states[7], label='O', marker="^")
# plt.plot(time_points, final_states[3], label='N2+', marker="x")
# plt.plot(time_points, final_states[4], label='N+', marker="d")
# plt.plot(time_points, final_states[5], label='O2+', marker="v")
# plt.plot(time_points, final_states[8], label='O+', marker="*")
# plt.xlabel('Time [s]')
# plt.ylabel(r'Density [m$^{-3}$]')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Species Densities Over Time at Power '+str(power)+' W')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"images/Species_densities_over_time.png")

# df_temporal = pd.DataFrame(final_states.T, columns=species.names)
# df_temporal['Time (s)'] = time_points
# df_temporal.to_csv(f"logs/temporal_evolution_NO_{power}W_alt_{altitude}km.csv", index=False)

# plt.figure(figsize=(7,5))
# plt.plot(time_points, final_states[species.nb], label='Electron Temp', marker="o")
# plt.plot(time_points, final_states[species.nb + 1], label='Monoatomic Neutral Temp', marker="s")
# plt.plot(time_points, final_states[species.nb + 2], label='Diatomic Neutral Temp', marker="^")
# plt.xlabel('Time [s]')
# plt.ylabel('Temperature [eV]')
# plt.title('Temperatures Over Time at Power '+str(power)+' W')
# plt.xscale('log')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"images/Temperatures_over_time.png")

# final_states = sol.y

# # Extract time points
# time_points = sol.t

# # Create figure and primary axis
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# # Plot species concentrations on the first subplot
# #ax1.yscale('log')
# for i, specie in enumerate(species.species):
#     ax1.semilogy(time_points, final_states[i], label=specie.name)
# ax1.set_ylabel('Density of species (m^-3)')
# ax1.legend(loc='best')
# ax1.grid()
#ax1.grid(which='both', axis='y')

# Create a secondary y-axis for Xenon temperature
# ax3 = ax2.twinx()

# # Plot temperatures: Electron on primary y-axis, Xenon on secondary y-axis
# ax2.plot(time_points, final_states[species.nb], label='Electron Temp (eV)', color='blue')
# for i in range(1,3):
#     ax3.plot(time_points, final_states[species.nb + i], linestyle='--', label= f"Molecules with {i} atoms Temp (eV)")

# ax2.set_ylabel('Electron Temperature', color='blue')
# ax3.set_ylabel('Molecules Temperature', color='red')
# ax2.tick_params(axis='y', labelcolor='blue')
# ax3.tick_params(axis='y', labelcolor='red')

# # Combine legends
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# lines_3, labels_3 = ax3.get_legend_handles_labels()
# ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc='best')

# # Set labels and title
# ax2.set_xlabel('Time (s)')
# ax1.set_title('Species Concentrations Over Time')
# ax2.set_title('Temperature Evolution')
# ax2.grid()

# # Show the plot
# plt.tight_layout()
# plt.show()


# sol = model.solve(0, 5e-1)    # TODO Needs some testing

# final_states = sol.y
# print(final_states)

# # Plot the results
# time_points = sol.t
# fig, ax1 = plt.subplots()

# # Plot species concentrations on the first y-axis
# for i, specie in enumerate(species_list.species):
#     ax1.plot(time_points, final_states[i], label=specie.name)
# ax1.set_ylabel('Concentration')
# ax1.tick_params(axis='y')

# # Create a second y-axis for temperature
# ax2 = ax1.twinx()
# for i in range(2):
#     ax2.plot(time_points, final_states[len(species_list.species) + i], label="Temp " + str(i), linestyle='--')
# ax2.set_ylabel('Temperature')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# # Combine legends from both axes
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, bbox_to_anchor=(0.5, -0.1), ncol=2)

# plt.xlabel('Time (s)')
# plt.ylabel('Concentration')
# plt.title('Species Concentrations Over Time')
# plt.legend()
# plt.grid()
# plt.show()