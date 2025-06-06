
import os
import pandas as pd

from class_utils import Equation

def load_eq(path_folder, idx, num_eqs_per_csv):
    index_file = str(int(idx / num_eqs_per_csv))
    eq = pd.read_csv(os.path.join(path_folder, f"{index_file}.csv"),
                     skiprows=idx - int(index_file) * int(num_eqs_per_csv), nrows=1).values
    eq = Equation(F_prefix=eq[0][0], G_prefix=eq[0][1], F_infix=eq[0][2], G_infix=eq[0][3], F_infix_c=None, G_infix_c=None, dimension=int(eq[0][4]),id=eq[0][5])
    
    return eq