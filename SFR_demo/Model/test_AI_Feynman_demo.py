import numpy as np
import hydra
import warnings
import os
import json
import pandas as pd
import torch
import numexpr as ne
import math
import matplotlib.pyplot as plt
import sympy as sp

from functools import partial
from NNS_metrics import cal_metrics
from model import Model
from class_utils import FitParams
from test_utils import FunctionEnvironment


def tree_to_numexpr_fn_F(F_eq,dim,n_points):
    def wrapped_numexpr_fn(x):
        F_local_dict={}
        for d in range(dim):
            F_local_dict["x_i_{}".format(d)] = x[:,:, d].reshape(-1,1)
        try:
            F_vals = ne.evaluate(F_eq, local_dict=F_local_dict).reshape(n_points, -1, 1)
            vals = torch.from_numpy(F_vals)
        except Exception as e:
            print(e)
            return None
        return vals
    return wrapped_numexpr_fn


@hydra.main(version_base=None, config_path="config", config_name="AI_Feynman")
def test(cfg):
    with open(os.path.join(cfg.metadata_path,"metadata.json")) as f:
        metadata=json.load(f)
    params_fit=FitParams(word2id=metadata["word2id"],
                         id2word=metadata["id2word"],
                         total_variables=metadata["variables"])
    env=FunctionEnvironment(cfg)
    model = Model.load_from_checkpoint(cfg.model_path, cfg=cfg, metadata=metadata,map_location="cuda")
    model.eval()
    model.cuda()
    fitfunc = partial(model.fitfunc_F_trans, cfg_params=[cfg.beam_size, params_fit, env])
    
    print("######TEST Setting######")
    print("DATASET:AI_Feynman")

    Fs_info=pd.read_csv(f"SFR/data/{cfg.testdata_path}.csv")[0:1].values[:,:]
    rows={"F_predict":[],"F_true":[],
        "R_Close_01":[],"R_Close_001":[],"R_Close_0001":[],"R_R2":[],
        "P_Close_01":[],"P_Close_001":[],"P_Close_0001":[],"P_R2":[]}
    for count,F_info in enumerate(Fs_info):
        print(count)
        n_points=cfg.n_point
        F_eq=F_info[1]
        dim=min(F_info[2],3)
        _X=np.random.normal(0,1,(n_points,1,dim))
        tree=tree_to_numexpr_fn_F(F_eq,dim,n_points)
        std=2
        mean=0
        X=_X*std+mean
        y=tree(X)
        _X=torch.from_numpy(_X)


        num_sample_state=cfg.sample_state
        num_sample_node=cfg.sample_node
        num_sample_time_per_node=math.ceil(num_sample_state/num_sample_node)
        _index_sample_nodes = [0]
        index_sample_time = []
        index_sample_node = []
        for ii in range(num_sample_node):
            index_sample_time.append(torch.from_numpy(np.random.choice(a=list(range(n_points)), size=num_sample_time_per_node, replace=False)).long())
            index_sample_node.append(torch.Tensor([_index_sample_nodes[ii]] * num_sample_time_per_node))
        index_sample_time = torch.cat(index_sample_time, dim=-1).long().view(-1)[:num_sample_state]
        index_sample_node = torch.cat(index_sample_node, dim=-1).long().view(-1)[:num_sample_state]


        nodes_state=_X[index_sample_time, index_sample_node, :]
        y = y[index_sample_time, index_sample_node, :]
        
        if y is None:
            continue
        MAX_FLOAT_VALUE=1e38
        MIN_FLOAT_VALUE=-1e38
        y_test=y.numpy()
        if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)) or np.any(y_test > MAX_FLOAT_VALUE) or np.any(y_test < MIN_FLOAT_VALUE):
            continue

        nodes_state=torch.nn.functional.pad(nodes_state,(0,3-dim),value=0)
        nodes_state=nodes_state.unsqueeze(0)
        y=y.unsqueeze(0)
        result,R_metrics_result = fitfunc(nodes_state,y,dim)
        F_regression=str(result['best_bfgs_preds'][0]).replace("x_i_0",f"((x_i_0-{mean})/{std})").replace("x_i_1",f"((x_i_1-{mean})/{std})").replace("x_i_2",f"((x_i_2-{mean})/{std})") 
        F_regression=sp.simplify(F_regression)
        F_true=sp.simplify(F_eq)

        Out_Domain_X=np.random.normal(0,10,(cfg.Out_Domain_point,1,dim)) 

        Out_Domain_tree=tree_to_numexpr_fn_F(str(F_regression),dim,cfg.Out_Domain_point)
        Out_Domain_y=Out_Domain_tree(Out_Domain_X).reshape(-1).numpy()

        True_tree=tree_to_numexpr_fn_F(str(F_true),dim,cfg.Out_Domain_point) 
        True_y=True_tree(Out_Domain_X).reshape(-1).numpy()
        P_metrics_result=cal_metrics(Out_Domain_y,True_y)

        rows["F_predict"].append(str(F_regression))
        rows["F_true"].append(str(F_true))

        rows["R_Close_01"].append(R_metrics_result['close_0.1'])
        rows["R_Close_001"].append(R_metrics_result['close_0.01'])
        rows["R_Close_0001"].append(R_metrics_result['close_0.001'])
        rows["R_R2"].append(R_metrics_result['R^2'])
        rows["P_Close_01"].append(P_metrics_result['close_0.1'])
        rows["P_Close_001"].append(P_metrics_result['close_0.01'])
        rows["P_Close_0001"].append(P_metrics_result['close_0.001'])
        rows["P_R2"].append(P_metrics_result['R^2'])

        if cfg.Draw:
            if dim==1:
                fig,ax=plt.subplots(figsize=(10,7))
                ax.scatter(Out_Domain_X.reshape(-1),Out_Domain_y,c="r",label="SFR",alpha=0.3)
                ax.scatter(Out_Domain_X.reshape(-1),True_y,c="b",label="True",s=5)
                ax.set_xlabel(r"$x_{i,0}$",fontsize=20)
                ax.set_ylabel(r"$y$",fontsize=20)
                ax.tick_params('both',labelsize=20)
                ax.legend(loc="upper right",prop={'size':20})
                ax.set_title("True:"+str(F_true)+"\n"+"SFR:"+str(F_regression))
                plt.savefig(f"SFR/results/Figure_AI_Feynman_{count}.png",bbox_inches="tight")

    pd.DataFrame(rows).to_csv(f"SFR/results/{cfg.testdata_path}_result.csv")
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    test()