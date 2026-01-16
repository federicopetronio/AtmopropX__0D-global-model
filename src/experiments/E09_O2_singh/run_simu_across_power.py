import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from pathlib import Path
import pandas as pd
import os 
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
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
from reaction_set_O import get_species_and_reactions

log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("logs")
outputs_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("outputs")

power_list_W = [50, 100, 150, 200, 250, 300]  # in W
power_list_W = range(25, 351, 25)  # in W
# power_list_W = [ 100,  200,  300]  # in W

chamber = Chamber(config_dict)
final_states_list = []

density_vec = []
temperature_vec = []
power_list_W_vec = []

for target_power in power_list_W:
    chamber.target_power = target_power
    print(chamber.target_power)
    species, initial_state, reactions_list, electron_heating, modifier_func = get_species_and_reactions(chamber)
    model = GlobalModel(species, reactions_list, chamber, electron_heating, simulation_name=f"O2_Thorsteinsonn_{target_power:.3f}", log_folder_path=log_folder_path)


    # Solve the model
    try:
        print("Solving model... for power ", target_power)
        print(species.names)
        with HiddenPrints():
            sol = model.solve(0, 1, initial_state, (modifier_func, None))  # TODO Needs some testing
        final_states_list.append(list(sol.y[:, -1])+[target_power])
        # print(final_states_list[-1])
        density_vec.append(final_states_list[-1][species.names.index("e")])
        temperature_vec.append(final_states_list[-1][-4])  # T_e is the 4th last element
        power_list_W_vec.append(target_power)
        print("Model resolved !")
    except Exception as exception:
        print("Entering exception...")
        model.var_tracker.save_tracked_variables()
        print("Variables saved")
        raise exception


print(final_states_list)
final_states_df = pd.DataFrame(final_states_list, columns=species.names+["T_e", "T_mono", "T_diato", "target_power_W"])
final_states_df.to_csv(outputs_folder_path.joinpath("new_O2_singh_final_states_across_power_with_thermal_diff.csv"))
print(density_vec)
print(temperature_vec)
print(power_list_W_vec)
df1 = pd.DataFrame({
    "Power_W": power_list_W_vec,
    "Electron_density_m3": density_vec,
    "Electron_temperature_eV": temperature_vec
})
df1.to_csv(outputs_folder_path.joinpath("new_O2_singh_e_density_Te_across_power_with_thermal_diff.csv"))
