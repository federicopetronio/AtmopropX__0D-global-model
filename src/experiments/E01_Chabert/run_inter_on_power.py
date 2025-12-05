import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from pathlib import Path

# If global_model_package is not installed as package with pip install -e . , adds the global_model_package to the path so that it can be imported as a package
try :
    import global_model_package
    print("'global_model_package' imported as pip package or already in sys.path.")
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber

from config import config_dict
from reaction_set_Xe import get_species_and_reactions



chamber = Chamber(config_dict)

density_ne = []
density_ng = []
Thrust_ni = []
Thrust_ng = []
Temp_e = []
Temp_ng = []


# Solve the model
power_list = np.linspace(50, 1600, 30)


for power in power_list:
    species, initial_state, reactions_list, electron_heating = get_species_and_reactions(chamber, power=power)
    log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("logs")
    model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name="Chabert_reproduction", log_folder_path=log_folder_path)

    # Solve the model
    try:
        print("Solving model...")
        sol = model.solve(0, 1, initial_state)  # TODO Needs some testing
        print("Model resolved !")
    except Exception as exception:
        print("Entering exception...")
        model.var_tracker.save_tracked_variables()
        print("Variables saved")
        raise exception
    final_states = sol.y
    density_ne.append(final_states[0][-1])
    density_ng.append(final_states[1][-1])
    Temp_e.append(final_states[species.nb][-1])
    Temp_ng.append(final_states[species.nb + 1][-1])
    h_L = chamber.h_L(final_states[1][-1])

    Thrust_ni.append(h_L * density_ne[-1] * (e * Temp_e[-1])**.5 * (2 * e * chamber.V_grid)**.5 * chamber.R**2. * pi * chamber.beta_i)
    print("H_L =", h_L)
    print("density_ne =", density_ne[-1])
    print("Temp_e =", (e * Temp_e[-1])**.5, k, e, Temp_e[-1])
    print("(2 * e * chamber.V_grid)**.5 =", (2 * e * chamber.V_grid)**.5)
    print( "chamber.R**2. * pi * chamber.beta_i =", chamber.R**2. * pi, chamber.beta_i)
    # n_density
    n_density = final_states[:species.nb]

# Extract time points
time_points = sol.t

# Create figure and primary axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Plot species concentrations on the first subplot
#ax1.yscale('log')
for i, specie in enumerate(species.species):
    ax1.semilogy(time_points, final_states[i], label=specie.name)
ax1.set_ylabel('Density of species (m^-3)')
ax1.legend(loc='best')
ax1.grid()
#ax1.grid(which='both', axis='y')

# Create a secondary y-axis for Xenon temperature
ax3 = ax2.twinx()

# Plot temperatures: Electron on primary y-axis, Xenon on secondary y-axis
ax2.plot(time_points, final_states[species.nb], label='Electron Temp (eV)', color='blue')
for i in range(1,3):
    ax3.plot(time_points, final_states[species.nb + i], linestyle='--', label= f"Molecules with {i} atoms Temp (eV)")

ax2.set_ylabel('Electron Temperature', color='blue')
ax3.set_ylabel('Molecules Temperature', color='red')
ax2.tick_params(axis='y', labelcolor='blue')
ax3.tick_params(axis='y', labelcolor='red')

# Combine legends
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc='best')

# Set labels and title
ax2.set_xlabel('Time (s)')
ax1.set_title('Species Concentrations Over Time')
ax2.set_title('Temperature Evolution')
ax2.grid()

# Show the plot
plt.tight_layout()
plt.savefig(f"Chabert_reproduction_power_{power}W.png")

plt.figure()
plt.semilogy(power_list, density_ne, label='Electron Density')
plt.semilogy(power_list, density_ng, label='Neutral Gas Density')
plt.xlabel('RF Power (W)')
plt.ylabel('Density (m^-3)')
plt.title('Densities vs RF Power')
plt.legend()
plt.grid()
plt.ylim(1e16, 1e20)
plt.savefig(f"Chabert_reproduction_densities_vs_power.png")

plt.figure()
plt.semilogy(power_list, np.array(Thrust_ni)*1.e3, label='Ion Thrust Estimate')
plt.xlabel('RF Power (W)')
plt.ylabel('Thrust (N)')
plt.title('Ion Thrust Estimate vs RF Power')
plt.legend()
plt.grid()
plt.ylim(1e-1, 100)
plt.savefig(f"Chabert_reproduction_thrust_vs_power.png")

plt.figure()
plt.subplot(211)
plt.plot(power_list, Temp_e, label='Electron Temperature')
plt.ylabel('Electron Temperature (eV)')    
plt.grid()
plt.xlim(0, 1600)
plt.ylim(2, 6)
plt.subplot(212)
plt.plot(power_list, np.array(Temp_ng) * e / k, label='Neutral Gas Temperature')
plt.scatter(power_list[-1], np.array(3.44e-2) * e / k, label='Neutral Gas Temperature')
plt.xlabel('RF Power (W)')
plt.ylabel('Neutral Gas Temperature (K)')
plt.title('Temperatures vs RF Power')
plt.xlim(0, 1600)
plt.ylim(200, 600)
plt.grid()
plt.tight_layout()
plt.savefig(f"Chabert_reproduction_temperatures_vs_power.png")


print(final_states[0][-1], 2. , e ,chamber.V_grid ,chamber.R**2. ,pi, 0.7)
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