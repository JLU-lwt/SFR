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
from bfgs import bfgs_FG_real_isomorphism
from model import Model
from class_utils import FitParams
from test_utils import FunctionEnvironment
def tree_to_numexpr_fn_real(F_eq,G_eq,dim,n_points,row,col,weight):
    def wrapped_numexpr_fn(x):


        F_local_dict={}
        G_local_dict={}
        F_local_dict["x_i_0"] = x[:,:].reshape(-1,1)
        G_local_dict["x_i_0"] = x[:,col].reshape(-1,1)
        G_local_dict["x_j_0"] = x[:,row].reshape(-1,1)
        
        F_vals=ne.evaluate(F_eq,local_dict=F_local_dict).reshape(n_points,-1)
        G_vals=ne.evaluate(G_eq,local_dict=G_local_dict).reshape(n_points,-1)

        G_vals=np.multiply(G_vals,weight).astype(float)



        vals = torch.scatter_add(torch.from_numpy(F_vals), 1, col.repeat(n_points).reshape(n_points,-1), torch.from_numpy(G_vals))
        return np.array(vals)
    return wrapped_numexpr_fn

def sampling_strategy_real(data, num_sampling_time,clusters,num_sampling_per_cluster,num):

    data=data.squeeze(-1).permute(1,0)
    sample_node = [num]
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


@hydra.main(version_base=None, config_path="config", config_name="Real")
def test(cfg):
    with open(os.path.join(cfg.metadata_path,"metadata.json")) as f:
        metadata=json.load(f)
    params_fit=FitParams(word2id=metadata["word2id"],
                         id2word=metadata["id2word"],
                         total_variables=metadata["variables"])
    env=FunctionEnvironment(cfg)
    model = Model.load_from_checkpoint(cfg.model_path, cfg=cfg, metadata=metadata,map_location="cuda:0")
    model.eval()
    model.cuda()
    fitfunc = partial(model.fitfunc_FG_Dy_real, cfg_params=[cfg.beam_size, params_fit, env])
    print("######TEST Setting######")
    print("DATASET:Real")

    if cfg.metric_name=="H1N1":
        dim=1
        period=45
        limit=100
        w=0.0014
    
    raw_data=pd.read_excel(f"SFR/data/Real_demo.xlsx",sheet_name=f"{cfg.metric_name}",header=None).values
    raw_dx=raw_data[1:,:]
    test_data=torch.from_numpy(raw_data[:-1,:]).unsqueeze(-1).float()
    diff_X=torch.from_numpy(raw_dx).unsqueeze(-1).float()

    country_select_index=[]
    time_select_index=[]
    for country_index in range(raw_data.shape[1]):
        first_no0_index=(np.nonzero(raw_data[:,country_index])[0])[0]
        try:
            if len(raw_data[first_no0_index:,country_index])>=period and (raw_data[first_no0_index:,country_index])[period-1]>=limit:
                country_select_index.append(country_index)
                time_select_index.append([first_no0_index,first_no0_index+period-1])
        except Exception as e:
            continue


    raw_topo_data=pd.read_csv(f"SFR/data/Real_demo_{cfg.metric_name}_topo.csv",header=None).values[:,1:]
    raw_topo_data=raw_topo_data*w

    topo_data=raw_topo_data.copy()
    topo_data[topo_data>0]=1

    
    row=[]
    col=[]

    for i in range(topo_data.shape[0]):
        for j in range(topo_data.shape[1]):
            if topo_data[i][j]==1:
                row.append(j)
                col.append(i)
    
    row=torch.LongTensor(row)
    col=torch.LongTensor(col)

    
    topo_scale=len(country_select_index)
    if cfg.metric_name=="H1N1" or cfg.metric_name=="Sars":
        country_select_index=country_select_index
    
    elif cfg.metric_name=="COVID19":
        country_select_index=country_select_index[:20]

    print(country_select_index)
    rows={"beam":[],"Node":[],"F_predict":[],"G_predict":[],"Close0.01":[],"Close0.001":[],"R2":[],"MSE":[],"MAPE":[]}
    index_sample_time=[]
    index_sample_node=[]
    for num,ii in enumerate(country_select_index):
        test_data_sample=test_data[time_select_index[num][0]:time_select_index[num][1],:,:]
        sample_node, sample_time,mean,std=sampling_strategy_real(test_data_sample, num_sampling_time=cfg.T, clusters=cfg.clusters, num_sampling_per_cluster=cfg.num_sampling_per_cluster,num=ii)
        sample_time=sample_time+time_select_index[num][0]
        per_index_sample_time=torch.from_numpy(sample_time.reshape(-1)).long()
        per_index_sample_node = torch.Tensor([sample_node[0]] * cfg.T)

        index_sample_time.append(per_index_sample_time)
        index_sample_node.append(per_index_sample_node)

    
    index_sample_time=torch.cat(index_sample_time, dim=-1).long().view(-1)
    index_sample_node = torch.cat(index_sample_node, dim=-1).long().view(-1)


    nodes_state=test_data[index_sample_time, index_sample_node, :] 
    _nodes_neighbours=test_data[:,row,:]
    _nodes_neighbours_info=torch.ones_like(_nodes_neighbours[:,:,0])*-1
    
    for j in range(len(index_sample_node)):
        _nodes_neighbours_info[index_sample_time[j],col==index_sample_node[j]]=j
    nodes_neighbours=_nodes_neighbours[_nodes_neighbours_info>=0,:]
    nodes_neighbours_info=_nodes_neighbours_info[_nodes_neighbours_info>=0]
    y = diff_X[index_sample_time, index_sample_node, :]





    X_mean=torch.mean(torch.cat([nodes_state.reshape(-1),nodes_neighbours.reshape(-1)],dim=0))
    X_std=torch.std(torch.cat([nodes_state.reshape(-1),nodes_neighbours.reshape(-1)],dim=0))



    nodes_state_new=(nodes_state-X_mean)/X_std
    nodes_neighbours_new=(nodes_neighbours-X_mean)/X_std
    nodes_state_new_padding=torch.nn.functional.pad(nodes_state_new,(0,3-dim),value=0).unsqueeze(0)
    nodes_neighbours_new_padding=torch.nn.functional.pad(nodes_neighbours_new,(0,3-dim),value=0).unsqueeze(0)
    nodes_neighbours_info_padding=nodes_neighbours_info.unsqueeze(0)
    y_padding=y.unsqueeze(0)

    F_beam,G_beam = fitfunc(nodes_state_new_padding,nodes_neighbours_new_padding,nodes_neighbours_info_padding,y_padding,dim)
    B=0
    for F_ww,G_ww in zip(F_beam,G_beam):
        R2_m=0
        results=bfgs_FG_real_isomorphism(F_ww,G_ww,nodes_state_new_padding,nodes_neighbours_new_padding,nodes_neighbours_info_padding,y_padding,dim,params_fit,env,raw_topo_data,index_sample_node)
        print(results)
        F_predict_raw=str(results[0][0])
        G_predict_raw=str(results[0][1])
        F_predict=F_predict_raw.replace(f"x_i_{0}",f"((x_i_{0}-{X_mean.item()})/{X_std.item()})")
        G_predict=G_predict_raw.replace(f"x_i_{0}",f"((x_i_{0}-{X_mean.item()})/{X_std.item()})").replace(f"x_j_{0}",f"((x_j_{0}-{X_mean.item()})/{X_std.item()})")
        F_predict=str(sp.sympify(F_predict))
        G_predict=str(sp.sympify(G_predict))

        F_regression_isomorphism=F_predict
        G_regression_isomorphism=G_predict
        print(F_regression_isomorphism)
        print(G_regression_isomorphism)
        for num,i in enumerate(country_select_index):
            row_select=[]
            col_select=[]
            raw_topo_select=[]
            for index,e in enumerate(col):
                if e==i:
                    row_select.append(row[index])
                    col_select.append(col[index])
                    raw_topo_select.append(raw_topo_data[col[index]][row[index]])
            row_select=torch.LongTensor(row_select)
            col_select=torch.LongTensor(col_select)

            metric_data=np.array(test_data[time_select_index[num][0]:time_select_index[num][1],:].squeeze(-1))

            weight=np.repeat(np.array(raw_topo_select).reshape(1,-1),metric_data.shape[0],0)
            pre_tree=tree_to_numexpr_fn_real(F_regression_isomorphism,G_regression_isomorphism,dim,metric_data.shape[0],row_select,col_select,weight)
            pre_y=pre_tree(metric_data)

            sum_value=metric_data[0,i]
            draw_y=[sum_value]
            for j in range(period-1):
                sum_value=pre_y[j,i]
                draw_y.append(sum_value)


            metric_result=cal_metrics(np.array(draw_y),np.array(raw_data[time_select_index[num][0]:time_select_index[num][1]+1,i]))

            rows["beam"].append(B)
            rows["Node"].append(num)
            rows["F_predict"].append(str(F_regression_isomorphism))
            rows["G_predict"].append(str(G_regression_isomorphism))
            rows["Close0.01"].append(metric_result['close_0.01'])
            rows["Close0.001"].append(metric_result['close_0.001'])
            rows["R2"].append(metric_result['R^2'])
            rows["MSE"].append(metric_result['MSE'])
            rows["MAPE"].append(metric_result['MAPE'])

        B=B+1
        pd.DataFrame(rows).to_csv(f"SFR/results/Real_Demo_{cfg.metric_name}.csv",index=False)
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    test()