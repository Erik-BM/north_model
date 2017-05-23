from north_model_class import north_model

import numpy as np
import scipy as sp

import string
import random    

def id_generator(size=11, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


def save_files(inst, directory, scen = None):
    grid = 400
    if scen == None:
        scen = id_generator()

    arc = np.sin(64 * np.pi / 180)

    temp = np.mean(inst.temp,axis=1)
    deep_temp = np.mean(inst.deep_temp,axis=1)

    arctic_temp = np.mean(inst.temp[:, int(arc * grid):-1], axis=1)

    arctic_deep_temp = np.mean(inst.deep_temp[:, int(arc * grid):-1], axis=1)

    pole_temp = inst.temp[:,-1]

    name = directory + "pole_temp_" + scen + ".csv"
    np.savetxt(name, pole_temp, delimiter=',')

    name = directory + "temp_" + scen + ".csv"
    np.savetxt(name, temp, delimiter=',')
    
    name = directory + "deep_temp_" + scen + ".csv"
    np.savetxt(name, deep_temp, delimiter=',')

    name = directory + "ice_edge_" + scen + ".csv"
    np.savetxt(name, inst.ice_edge, delimiter=',')

    name = directory + "arctic_temp_" + scen + ".csv"
    np.savetxt(name, arctic_temp, delimiter=',')    

    name = directory + "arctic_deep_temp_" + scen + ".csv"
    np.savetxt(name, arctic_deep_temp, delimiter=',')      

f = np.zeros(1000) #one value per year
inst = model(grid_size = 400, forcing = f, load_from_init_file = False)
inst.run_model()
save_files(inst = inst, directory = "", scen = "test")  

print("Done")
quit()
