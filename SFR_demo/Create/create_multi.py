import hydra
import numpy as np
import os
import pandas as pd
import json


from mpi4py import MPI
from pathlib import Path
from dataclasses import dataclass

from environment import FunctionEnvironment

class CSVFilesCreator():
    def __init__(self, target_path: Path = None):
        target_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.target_path = target_path

    def create_single_csv_from_eqs(self, filename, rows):
        pd.DataFrame(rows).to_csv(os.path.join(self.target_path, str(filename) + ".csv"), index=False)
        print("Saving csv to", os.path.join(self.target_path, str(filename) + ".csv"))

@dataclass
class DatasetDetails:
    variables: list
    word2id: dict
    id2word: dict
    total_number_of_eqs_train: int
    eqs_per_csv_train: int
    total_number_of_eqs_val: int
    eqs_per_csv_val: int
    total_number_of_eqs_test: int
    eqs_per_csv_test: int


@hydra.main(config_name="config", version_base='1.2', config_path='config')


def creator(cfg):
    env=FunctionEnvironment(cfg)
    env.rng=np.random.RandomState()
    data_path = Path(f"datasets/{cfg.num_file_per_core * cfg.num_skeleton_per_file * size}_{cfg.random_label}/train")
    csv_creator = CSVFilesCreator(target_path=data_path)

    if rank==0:
        dataset_info=DatasetDetails(
            variables=list(env.generator.variables),
            id2word=env.id2word,
            word2id=env.word2id,
            total_number_of_eqs_train=cfg.num_file_per_core * cfg.num_skeleton_per_file * size,
            eqs_per_csv_train=cfg.num_skeleton_per_file,
            total_number_of_eqs_val=0,
            eqs_per_csv_val=0,
             total_number_of_eqs_test=0,
            eqs_per_csv_test=0
        )

        dataset_info_json=json.dumps(dataset_info.__dict__)
        with open (f"datasets/{cfg.num_file_per_core * cfg.num_skeleton_per_file * size}_{cfg.random_label}/metadata.json","w") as file:
            file.write(dataset_info_json)


    for i in range(cfg.num_file_per_core):
        count=0
        rows={"F_ske_prefix": [], "G_ske_prefix": [], "F_ske_infix": [], "G_ske_infix": [], "dim": [],"id":[]}
        while count<cfg.num_skeleton_per_file:
            F_eq,G_eq,dimension=env.gen_expr()

            
            if F_eq is not None or G_eq is not None:
                count+=1
                rows["F_ske_prefix"].append(F_eq.prefix(True))
                rows["G_ske_prefix"].append(G_eq.prefix(True))
                rows["F_ske_infix"].append(F_eq.infix(True))
                rows["G_ske_infix"].append(G_eq.infix(True))
                rows["dim"].append(dimension)
                rows["id"].append(str(i)+"_"+str(count))

        csv_creator.create_single_csv_from_eqs(rank*cfg.num_file_per_core+i,rows)
    

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    creator()

    