import numpy as np
import hydra
import warnings
import os
import json
import pandas as pd
import torch
import numexpr as ne

import matplotlib.pyplot as plt
import sympy as sp
import scipy

from functools import partial
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


from NNS_metrics import cal_metrics
from model import Model
from class_utils import FitParams
from test_utils import FunctionEnvironment




def finite_difference(X, delta_t):
    diff_X = (X[0:-4] - 8*X[1:-3] + 8*X[3:-1] - X[4:])/(12.0*delta_t)
    return X[2:-2], diff_X


def tree_to_numexpr_fn(F_tree,G_tree,row,col,dimension):
    F_infix = F_tree
    G_infix=G_tree
    def wrapped_numexpr_fn(x, t):
        F_local_dict={}
        G_local_dict={}
        for d in range(dimension):
            F_local_dict["x_i_{}".format(d)] = x[:, d]
            G_local_dict["x_i_{}".format(d)] = x[col, d]
            G_local_dict["x_j_{}".format(d)] = x[row, d]
        
        F_vals = ne.evaluate(F_infix, local_dict=F_local_dict).reshape(-1, 1)
        G_vals = ne.evaluate(G_infix, local_dict=G_local_dict).reshape(-1, 1)
        vals = torch.scatter_add(torch.from_numpy(F_vals), 0, col.view(-1, 1).repeat(1,dimension), torch.from_numpy(G_vals)).numpy()


        return vals
    return wrapped_numexpr_fn

def integrate_ode(y0, times, F_tree, G_tree,row,col, dimension):
    tree = tree_to_numexpr_fn(F_tree,G_tree,row,col,dimension)
    
    
    def func(y, t):
        return tree(np.array(y.reshape(-1, dimension)), t).reshape(-1)
    trajectory = scipy.integrate.odeint(func,y0.reshape(-1),times,rtol=1e-9, atol=1e-9)
    return trajectory


def sampling_strategy_new(data, topo, num_sampling_node, num_sampling_time,clusters,num_sampling_per_cluster):

    data=data.squeeze(-1).permute(1,0)
    row, col = topo
    node_indice, node_inDegree = torch.unique(col, return_counts=True)
    sample_N_num = num_sampling_node
    sample_node = np.random.choice(a=node_indice, size=sample_N_num, replace=False,p=np.array(node_inDegree / torch.sum(node_inDegree)))
    
    sample_data = data[sample_node, :]

    n_clusters = clusters
    num_sampling_per_cluster = num_sampling_per_cluster
    sampling_time_idx=[]
    time_clip=n_clusters*num_sampling_per_cluster

    for i in range(sample_data.shape[0]):
        time_per_node_data=sample_data[i,:].unsqueeze(-1)
        gm = GaussianMixture(n_components=n_clusters, init_params='k-means++', random_state=0).fit(time_per_node_data)
        labels = gm.predict(time_per_node_data)
        
        sampling_data_idx = []
        for j in range(n_clusters):
            sampling_data_j_idx = np.random.choice(np.linspace(0, len(labels) - 1, len(labels), dtype=np.int32)[labels == j],
                                                min(num_sampling_per_cluster, len(time_per_node_data[labels == j])), replace=False)
            
            sampling_data_idx = sampling_data_idx + sampling_data_j_idx.tolist()
        
        time_sample_num=len(sampling_data_idx)
        time_clip=min(time_sample_num,time_clip)

        sample_time_per_node_idx=torch.Tensor(sampling_data_idx)
        sample_time_per_node_idx=torch.nn.functional.pad(sample_time_per_node_idx,(0,n_clusters*num_sampling_per_cluster-time_sample_num))
        sampling_time_idx.append(sample_time_per_node_idx.unsqueeze(0))
    

    sampling_time_idx=torch.cat(sampling_time_idx,dim=0)[:,:time_clip].long()

    sample_data_decor=torch.cat([sample_data[i][sampling_time_idx[i]].unsqueeze(0) for i in range(sample_data.size(0))],dim=0)

    mean=torch.mean(sample_data_decor)
    std_dev=torch.std(sample_data_decor)

    z_scores = (sample_data_decor - mean) / std_dev
    
    pp = norm.cdf(z_scores)
    p = pp / np.sum(pp)
    
    sample_T_num = num_sampling_time
    if time_clip<sample_T_num:
        sample_time =np.array([np.random.choice(a=sampling_time_idx[i], size=sample_T_num, replace=True, p=p[i]/np.sum(p[i])) for i in range(len(sample_node))])
    else:
        sample_time =np.array([np.random.choice(a=sampling_time_idx[i], size=sample_T_num, replace=False, p=p[i]/np.sum(p[i])) for i in range(len(sample_node))])

    return sample_node, sample_time, mean,std_dev  

@hydra.main(version_base=None, config_path="config", config_name="Dynamics")
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
    fitfunc = partial(model.fitfunc_FG_Dy, cfg_params=[cfg.beam_size, params_fit, env])

    print("######TEST Setting######")
    print("DATASET:Dynamcis")

    if cfg.metric_name=="LV":
        sample_clip=1000
        delta_t=0.0001
        F_eq="0.5*x_i_0-x_i_0**2"
        G_eq="-x_j_0*x_i_0"
        t_start=0.0
        t_end=0.5
        t_inc=0.0001
        dim=1
    


    raw_data=pd.read_csv(f"SFR/data/Dynamics_demo_{cfg.metric_name}_{cfg.topo_type}_data.csv").values[1:,:]
    test_data=torch.from_numpy(raw_data)[:sample_clip,:].unsqueeze(-1)

    test_data,diff_X=finite_difference(test_data,delta_t)

    topo_info = pd.read_csv(f"SFR/data/Dynamics_demo_{cfg.metric_name}_{cfg.topo_type}_topo.csv",header=None).values
    sparse_adj=torch.LongTensor(topo_info)
    row, col = sparse_adj
    rows={"F_predict":[],"F_true":[],"G_predict":[],"G_true":[],
        "R_Close_01":[],"R_Close_001":[],"R_Close_0001":[],"R_R2":[],
        "P_Close_01":[],"P_Close_001":[],"P_Close_0001":[],"P_R2":[],"P_MAPE":[]}

    sample_node, sample_time,mean,std=sampling_strategy_new(test_data, [row,col], num_sampling_node=cfg.N, num_sampling_time=cfg.T,clusters=cfg.clusters,num_sampling_per_cluster=cfg.num_sampling_per_cluster)
    index_sample_time=torch.from_numpy(sample_time.reshape(-1)).long()
    
    index_sample_node = []
    for ii in range(cfg.N):
        index_sample_node.append(torch.Tensor([sample_node[ii]] * cfg.T))

    index_sample_node = torch.cat(index_sample_node, dim=-1).long().view(-1)
    nodes_state=test_data[index_sample_time, index_sample_node, :]
    _nodes_neighbours=test_data[:,row,:]
    _nodes_neighbours_info=torch.ones_like(_nodes_neighbours[:,:,0])*-1
    for j in range(len(index_sample_node)):
        _nodes_neighbours_info[index_sample_time[j],col==index_sample_node[j]]=j
    nodes_neighbours=_nodes_neighbours[_nodes_neighbours_info>=0,:]
    nodes_neighbours_info=_nodes_neighbours_info[_nodes_neighbours_info>=0]

    y = diff_X[index_sample_time, index_sample_node, :]
    ori_nodes_state=nodes_state.clone()
    ori_nodes_neighbours=nodes_neighbours.clone()

    X_mean=torch.mean(torch.cat([nodes_state.reshape(-1),nodes_neighbours.reshape(-1)],dim=0))
    X_std=torch.std(torch.cat([nodes_state.reshape(-1),nodes_neighbours.reshape(-1)],dim=0))

    nodes_state_new=(nodes_state-X_mean)/X_std

    nodes_neighbours_new=(nodes_neighbours-X_mean)/X_std
    nodes_state_new_padding=torch.nn.functional.pad(nodes_state_new,(0,3-dim),value=0).unsqueeze(0)
    nodes_neighbours_new_padding=torch.nn.functional.pad(nodes_neighbours_new,(0,3-dim),value=0).unsqueeze(0)


    nodes_neighbours_info_padding=nodes_neighbours_info.unsqueeze(0)
    y_padding=y.unsqueeze(0)

    
    result,metrics_result = fitfunc(nodes_state_new_padding,nodes_neighbours_new_padding,nodes_neighbours_info_padding,y_padding,dim)



    F_regression=str(result['best_bfgs_preds'][0])
    G_regression=str(result['best_bfgs_preds'][1])
            
    for i in range(dim):
        F_regression=F_regression.replace(f"x_i_{i}",f"((x_i_{i}-{X_mean.item()})/{X_std.item()})")
        G_regression=G_regression.replace(f"x_i_{i}",f"((x_i_{i}-{X_mean.item()})/{X_std.item()})")
        G_regression=G_regression.replace(f"x_j_{i}",f"((x_j_{i}-{X_mean.item()})/{X_std.item()})")

    

    F_regression=str(sp.simplify(F_regression))
    G_regression=str(sp.simplify(G_regression))

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    y0=pd.read_csv(f"SFR/data/Dynamics_demo_{cfg.metric_name}_{cfg.topo_type}_Inidata.csv",header=None).values[1:,:].T
    True_trajectory = integrate_ode(y0, t_range, F_eq, G_eq, row,col, dim)        
    True_trajectory=True_trajectory.T


    Predict_trajectory = integrate_ode(y0, t_range, F_regression, G_regression, row,col, dim)        
    Predict_trajectory=Predict_trajectory.T
    result=cal_metrics(True_trajectory,Predict_trajectory)

        



    rows["F_predict"].append(str(F_regression))
    rows["F_true"].append(str(F_eq))
    rows["G_predict"].append(str(G_regression))
    rows["G_true"].append(str(G_eq))
    rows["R_Close_01"].append(metrics_result['close_0.1'])
    rows["R_Close_001"].append(metrics_result['close_0.01'])
    rows["R_Close_0001"].append(metrics_result['close_0.001'])
    rows["R_R2"].append(metrics_result['R^2'])

    rows["P_Close_01"].append(result['close_0.1'])
    rows["P_Close_001"].append(result['close_0.01'])
    rows["P_Close_0001"].append(result['close_0.001'])
    rows["P_R2"].append(result['R^2'])
    rows["P_MAPE"].append(result['MAPE'])

    pd.DataFrame(rows).to_csv(f"SFR/results/Dynamics_demo_{cfg.metric_name}_{cfg.topo_type}_result.csv",index=False)

    if cfg.Draw:
        fig=plt.figure(figsize=(10,20))
        ax1=fig.add_subplot(211)
        ax2=fig.add_subplot(212)

        for i in range(True_trajectory.shape[0]):
            ax1.plot([n for n in range(True_trajectory.shape[1])],True_trajectory[i],linewidth=2)
            ax2.plot([n for n in range(Predict_trajectory.shape[1])],Predict_trajectory[i],linewidth=2)
        
        ax1.set_title("True:"+str(F_eq)+r"$+\sum$"+str(G_eq))
        ax1.set_xlabel(r"$t$",fontsize=20)
        ax1.set_ylabel(r"$x_{i,0}$",fontsize=20)
        ax1.set_xticks([0,1000,5000],["0",r"$T_{R}$",r"$T_{P}$"],fontsize=20)
        ax1.set_yticks([0,1.0,2.0,3.0,4.0],[0,1.0,2.0,3.0,4.0],fontsize=20)

        ax2.set_title("SFR:"+str(F_regression)+"\n"+r"$+\sum$"+str(G_regression))
        ax2.set_xlabel(r"$t$",fontsize=20)
        ax2.set_ylabel(r"$x_{i,0}$",fontsize=20)
        ax2.set_xticks([0,1000,5000],["0",r"$T_{R}$",r"$T_{P}$"],fontsize=20)
        ax2.set_yticks([0,1.0,2.0,3.0,4.0],[0,1.0,2.0,3.0,4.0],fontsize=20)
        plt.savefig(f"SFR/results/Figure_Dynamics_demo_{cfg.metric_name}_{cfg.topo_type}.png",bbox_inches="tight")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    test()