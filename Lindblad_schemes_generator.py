import itertools
import sympy as sp
import pickle
import time
import os
from multiprocessing import Pool

from  utils import *

def generate_Lindblad_maps(pool = None, path='Lindblad_maps/'):
    logfile_name = path + 'log.txt'
    i = 0
    s = Lindblad_scheme(logfile_name=logfile_name)
    os.makedirs(path, exist_ok=True)
    with open(path+f'scheme_{int(2*s.order)}.pkl', 'wb') as file:
        pickle.dump(s, file, protocol=pickle.HIGHEST_PROTOCOL)
    while True:
        try:   
            with open(path+f'scheme_{int(i)}.pkl', 'rb') as file:
                s = pickle.load(file)
                log_write(f"Scheme of order {i} was already saved", logfile_name=logfile_name)
            i += 1
        except FileNotFoundError as e:
            break
    try:
        with open(path+f'Lindblad_Kraus_ops_{int(2*s.order)}.pkl', 'rb') as file:
            Mmus = pickle.load(file)
        log_write(f"Lindblad Kraus operators of order {int(2*s.order)} was already saved", logfile_name=logfile_name)
    except FileNotFoundError as e:
        log_write(f"Computing Lindblad Kraus operators of order {int(2*s.order)}...", logfile_name=logfile_name)
        Mmus = Lindblad_Kraus_ops(s)
        log_write(f"Lindblad Kraus operators of order {int(2*s.order)} was saved", logfile_name=logfile_name)
        with open(path+f'Lindblad_Kraus_ops_{int(2*s.order)}.pkl', 'wb') as file:
            pickle.dump(Mmus, file, protocol=pickle.HIGHEST_PROTOCOL)
        log_write(f"Lindblad Kraus operators of order {int(2*s.order)} saved", logfile_name=logfile_name)

    log_write(f"Starting from scheme of order {int(2*s.order)+1}", logfile_name=logfile_name)
    while True:
        log_write(f"Computing scheme of order {int(2*s.order)+1}...", logfile_name=logfile_name)
        t0 = time.time()
        s.increase_order(pool=pool)
        log_write(f"Scheme of order {int(2*s.order)} computed in {round((time.time()-t0)/3600, 2)} hours, number of Lindblad Kraus operators: {len(s.Lindblad_states)}", logfile_name=logfile_name)
        with open(path+f'scheme_{int(2*s.order)}.pkl', 'wb') as file:
        # Dump data with highest protocol for best performance
            pickle.dump(s, file, protocol=pickle.HIGHEST_PROTOCOL)
        log_write(f"Computing Lindblad Kraus operators of order {int(2*s.order)}...", logfile_name=logfile_name)
        Mmus = Lindblad_Kraus_ops(s, pool=pool)
        with open(path+f'Lindblad_Kraus_ops_{int(2*s.order)}.pkl', 'wb') as file:
            pickle.dump(Mmus, file, protocol=pickle.HIGHEST_PROTOCOL)
        log_write(f"Lindblad Kraus operators of order {int(2*s.order)} saved", logfile_name=logfile_name)

    
if __name__ == "__main__":
    path  = 'Lindblad_maps/'
    os.makedirs(path, exist_ok=True)
    log_write(f"Running on {os.cpu_count()} CPU cores", init = False, logfile_name=path+'log.txt')
    p = Pool(processes=os.cpu_count())
    generate_Lindblad_maps(pool=p, path = path)

    