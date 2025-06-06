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
from topo_utils import Topo

def generate_float(infix,const_range):
    sign = np.random.choice([-1, 1])

    for i in range(infix.count("CONSTANT")):
        constant = np.random.uniform(np.log10(1/const_range), np.log10(const_range))
        constant = sign*10**constant
        infix=infix.replace("CONSTANT",str(np.around(constant,decimals=3)),1)

    return infix

def tree_to_numexpr_fn_FG(F_eq,G_eq,dim,n_points,row,col):
    def wrapped_numexpr_fn(x):
        F_local_dict={}
        G_local_dict={}
        for d in range(dim):
            F_local_dict["x_i_{}".format(d)] = x[:,:, d].reshape(-1,1)
            G_local_dict["x_i_{}".format(d)] = x[:,col, d].reshape(-1,1)
            G_local_dict["x_j_{}".format(d)] = x[:,row, d].reshape(-1,1)
        try:
            F_vals = ne.evaluate(F_eq, local_dict=F_local_dict).reshape(n_points, -1, 1)
            
            G_vals = ne.evaluate(G_eq, local_dict=G_local_dict).reshape(n_points, -1, 1)
            
            vals = torch.scatter_add(torch.from_numpy(F_vals), 1, col.repeat(n_points).reshape(n_points,-1,1), torch.from_numpy(G_vals))

        except Exception as e:
            
            return None
        return vals
    return wrapped_numexpr_fn


@hydra.main(version_base=None, config_path="config", config_name="USE")
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
    fitfunc = partial(model.fitfunc_FG, cfg_params=[cfg.beam_size, params_fit, env])
    
    print("######TEST Setting######")
    print("DATASET:USE")

    FGs_info=pd.read_csv(f"SFR/data/{cfg.testdata_path}.csv")[0:1].values[:,:]
    rows={"F_predict":[],"F_true":[],"G_predict":[],"G_true":[],
        "R_Close_01":[],"R_Close_001":[],"R_Close_0001":[],"R_R2":[],
        "P_Close_01":[],"P_Close_001":[],"P_Close_0001":[],"P_R2":[]}
    for count,FG_info in enumerate(FGs_info):
        print(count)

        F_eq=FG_info[0]
        G_eq=FG_info[1]
        dim=FG_info[2]
        F_eq=generate_float(F_eq,const_range=20)
        G_eq=generate_float(G_eq,const_range=20)
        topo_nodes = np.random.randint(cfg.minnode,cfg.maxnode)
        topo_type=np.random.choice(cfg.topo_type,1,False)
        topo = Topo(N=topo_nodes, topo_type=topo_type)
        topo_nodes = topo.N
        
        topo_nodes_have_neighbour=list(set(topo.sparse_adj[1].tolist()))
        row, col = topo.sparse_adj
        n_points=cfg.n_point

        _X=np.random.normal(0,1,(n_points,topo_nodes,dim))
        tree=tree_to_numexpr_fn_FG(F_eq,G_eq,dim,n_points,row,col)
        X=_X      
        y=tree(X)
        _X=torch.from_numpy(_X)

        num_sample_state=cfg.sample_state
        num_sample_nodes = np.random.randint(1, min(int(len(topo_nodes_have_neighbour) / 2), cfg.minnode))
        num_sample_time_per_node=math.ceil(num_sample_state/num_sample_nodes)
        _index_sample_nodes = np.random.choice(a=topo_nodes_have_neighbour, size=num_sample_nodes, replace=False)
        index_sample_time = []
        index_sample_node = []

        for ii in range(num_sample_nodes):
            if num_sample_time_per_node<num_sample_state:
                index_sample_time.append(torch.from_numpy(np.random.choice(a=list(range(n_points)), size=num_sample_time_per_node, replace=False)).long())
            else:
                index_sample_time.append(torch.from_numpy(np.random.choice(a=list(range(n_points)), size=num_sample_time_per_node, replace=True)).long())
            index_sample_node.append(torch.Tensor([_index_sample_nodes[ii]] * num_sample_time_per_node))
        

        index_sample_time = torch.cat(index_sample_time, dim=-1).long().view(-1)[:num_sample_state]
        index_sample_node = torch.cat(index_sample_node, dim=-1).long().view(-1)[:num_sample_state]

        nodes_state=_X[index_sample_time, index_sample_node, :]
        _nodes_neighbours=_X[:,row,:]
        _nodes_neighbours_info=torch.ones_like(_nodes_neighbours[:,:,0])*-1
        for j in range(len(index_sample_node)):
            _nodes_neighbours_info[index_sample_time[j],col==index_sample_node[j]]=j
    
        nodes_neighbours=_nodes_neighbours[_nodes_neighbours_info>=0,:]
        nodes_neighbours_info=_nodes_neighbours_info[_nodes_neighbours_info>=0]
        y = y[index_sample_time, index_sample_node, :]

        if y is None:
            continue
        
        MAX_FLOAT_VALUE=1e38
        MIN_FLOAT_VALUE=-1e38

        y_test=y.numpy()
        if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)) or np.any(y_test > MAX_FLOAT_VALUE) or np.any(y_test < MIN_FLOAT_VALUE):
            continue

        nodes_state=torch.nn.functional.pad(nodes_state,(0,3-dim),value=0)
        nodes_neighbours=torch.nn.functional.pad(nodes_neighbours,(0,3-dim),value=0)
        nodes_state=nodes_state.unsqueeze(0)
        nodes_neighbours=nodes_neighbours.unsqueeze(0)
        nodes_neighbours_info=nodes_neighbours_info.unsqueeze(0)
        y=y.unsqueeze(0)
        result,R_metrics_result = fitfunc(nodes_state,nodes_neighbours,nodes_neighbours_info,y,dim)

        F_regression=str(result['best_bfgs_preds'][0])
        G_regression=str(result['best_bfgs_preds'][1])

        F_regression=sp.simplify(F_regression)
        F_true=sp.simplify(F_eq)
        G_regression=sp.simplify(G_regression)
        G_true=sp.simplify(G_eq)


        Out_Domain_X=np.random.normal(0,10,(cfg.Out_Domain_point,topo_nodes,dim)) 
        Out_Domain_tree=tree_to_numexpr_fn_FG(str(F_regression),str(G_regression),dim,cfg.Out_Domain_point,row,col)
        Out_Domain_y=Out_Domain_tree(Out_Domain_X).numpy()

        True_tree=tree_to_numexpr_fn_FG(str(F_regression),str(G_regression),dim,cfg.Out_Domain_point,row,col) 
        True_y=True_tree(Out_Domain_X).numpy()
        P_metrics_result=cal_metrics(Out_Domain_y,True_y)

        rows["F_predict"].append(str(F_regression))
        rows["F_true"].append(str(F_true))
        rows["G_predict"].append(str(G_regression))
        rows["G_true"].append(str(G_true))

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
                fig,axes=plt.subplots(2,5,figsize=(20,7))
                selected_nodes=np.random.choice([n for n in range(topo_nodes)],10,False)
                for count,i in enumerate(selected_nodes):
                    axes[count//5,count%5].scatter(Out_Domain_X[:,i,:].reshape(-1),Out_Domain_y[:,i,:].reshape(-1),c="r",alpha=0.3,label="SFR")
                    axes[count//5,count%5].scatter(Out_Domain_X[:,i,:].reshape(-1),True_y[:,i,:].reshape(-1),c="b",s=5,label="True")
                    
                    axes[count//5,count%5].set_xlabel(r"$x_{i,0}$",fontsize=10)
                    axes[count//5,count%5].set_ylabel(r"$y$",fontsize=10)
                    axes[count//5,count%5].tick_params('both',labelsize=10)
                    axes[count//5,count%5].legend(loc="upper right")

                plt.suptitle("True:"+str(F_true)+r"$+\sum$"+str(G_true)+"\n"+"SFR:"+str(F_regression)+r"$+\sum$"+str(G_regression))
                plt.savefig(f"SFR/results/Figure_USE_{count}.png",bbox_inches="tight")
        
        pd.DataFrame(rows).to_csv(f"SFR/results/{cfg.testdata_path}_result.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    test()