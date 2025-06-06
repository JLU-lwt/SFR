import math
import random
import numpy as np
import pytorch_lightning as pl
import torch
import numexpr as ne
from functools import partial
from torch.utils import data
from class_utils import DataInfo, Equation
from load_utils import load_eq
from topo_utils import Topo

class EqDataset(data.Dataset):
    def __init__(self,dataset_path,eqs_per_csv,len):
        self.dataset_path=dataset_path
        self.eqs_per_csv=eqs_per_csv
        self.len=len
    
    def __getitem__(self, index):
        eq = load_eq(self.dataset_path, index, self.eqs_per_csv)
        return eq

    def __len__(self):
        return self.len



def tokenize(prefix_expr, word2id):
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr.split(","):
        tokenized_expr.append(word2id[i])
    tokenized_expr.append(word2id["F"])
    if len(tokenized_expr)>50:
        raise ValueError("too long")

    return tokenized_expr


def de_tokenize(tokenized_expr, id2word: dict):
    prefix_expr = []
    for i in tokenized_expr:
        if "F" == id2word[str(i)]:
            break
        else:
            prefix_expr.append(id2word[str(i)])
    return prefix_expr

def tokens_padding(tokens):
    max_len = max([len(y) for y in tokens])
    p_tokens = torch.zeros(len(tokens), max_len)
    for i, y in enumerate(tokens):
        y = torch.tensor(y).long()
        p_tokens[i, :] = torch.cat([y, torch.zeros(max_len - y.shape[0]).long()])
    return p_tokens

def tree_to_numexpr_fn(F_eq,G_eq,topo,dim,n_points):
    row,col=topo.sparse_adj
    
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

def generate_datapoints(F_eq,G_eq,dim,n_points,topo):
    
    MAX_FLOAT_VALUE=1e38
    MIN_FLOAT_VALUE=-1e38

    node_state=torch.randn(n_points,topo.N,dim)
    tree=tree_to_numexpr_fn(F_eq,G_eq,topo,dim,n_points)
    y=tree(np.array(node_state,dtype='float64'))
    
    if y is None:
        return None
    y_test=y.numpy()
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)) or np.any(y_test > MAX_FLOAT_VALUE) or np.any(y_test < MIN_FLOAT_VALUE):
        return None

            




    dataset={
       "data_node_state": node_state,
       "data_y": y
    }

    return dataset

def tree_to_numexpr_fn_only_F(F_eq,dim,n_points):


    
    def wrapped_numexpr_fn(x):
        F_local_dict={}
        
        for d in range(dim):
            F_local_dict["x_i_{}".format(d)] = x[:,:, d].reshape(-1,1)
            
        try:
            F_vals = ne.evaluate(F_eq, local_dict=F_local_dict).reshape(n_points, -1, 1)
            vals = torch.from_numpy(F_vals)
            
        except Exception as e:
            return None
        return vals
    return wrapped_numexpr_fn



def generate_datapoints_only_F(F_eq,dim,n_points):
    
    MAX_FLOAT_VALUE=1e38
    MIN_FLOAT_VALUE=-1e38



    node_state=torch.randn(n_points,1,dim)
    tree=tree_to_numexpr_fn_only_F(F_eq,dim,n_points)
    y=tree(np.array(node_state,dtype='float64'))
    if y is None:
        return None
    y_test=y.numpy()
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)) or np.any(y_test > MAX_FLOAT_VALUE) or np.any(y_test < MIN_FLOAT_VALUE):
        return None
    
    dataset={
       "data_node_state": node_state,
       "data_y": y
    }
    return dataset

def generate_float(infix,const_range):
    sign = np.random.choice([-1, 1])

    for i in range(infix.count("CONSTANT")):
        constant = np.random.uniform(np.log10(1/const_range), np.log10(const_range))
        constant = sign*10**constant
        infix=infix.replace("CONSTANT",str(constant),1)

    return infix


def evaluate_and_wrap(eqs_info, topo, index_sample_time, index_sample_node, cfg, meta,mode):
    
    nodes_state_batch = []
    nodes_neighbours_batch = []
    nodes_neighbours_info_batch = []
    y_batch = []
    f_token_batch = []
    g_token_batch = []
    f_equation_batch = []
    g_equation_batch = []
    f_ske_batch=[]
    g_ske_batch=[]
    dimension_batch = []

    real_eqs_info=[]
    


    for i in range(len(eqs_info)):
        for j in range(cfg.num_of_samples_eq_test):
            f_infix_c=generate_float(str(eqs_info[i].F_infix),cfg.constant_range)
            g_infix_c=generate_float(str(eqs_info[i].G_infix),cfg.constant_range)
            real_eqs_info.append(Equation(F_prefix=eqs_info[i].F_prefix,
                                          G_prefix=eqs_info[i].G_prefix,
                                          F_infix=eqs_info[i].F_infix,
                                          G_infix=eqs_info[i].G_infix,
                                          F_infix_c=f_infix_c,
                                          G_infix_c=g_infix_c,
                                          dimension=eqs_info[i].dimension,
                                          id=eqs_info[i].id+"_"+str(j)))
    
    random.shuffle(real_eqs_info)
    for eq_info in real_eqs_info:
        try:
            f_token=tokenize(str(eq_info.F_prefix), meta["word2id"])
            g_token=tokenize(str(eq_info.G_prefix), meta["word2id"])
        except ValueError:
            continue
        except KeyError:
            continue
        
        dimension= eq_info.dimension
        ns_data=generate_datapoints(eq_info.F_infix_c,eq_info.G_infix_c,dimension,cfg.times,topo)

        if ns_data is None:
            continue
    
        nodes_state_one = ns_data['data_node_state'][index_sample_time, index_sample_node, :]

        print(torch.mean(nodes_state_one),torch.std(nodes_state_one))


        nodes_state_one=torch.nn.functional.pad(nodes_state_one,(0,3-dimension),value=0)
        
        
        row, col = topo.sparse_adj
        _nodes_neighbours_one=ns_data["data_node_state"][:,row,:]
        _nodes_neighbours_info_one=torch.ones_like(_nodes_neighbours_one[:,:,0])*-1
        for j in range(len(index_sample_node)):
            _nodes_neighbours_info_one[index_sample_time[j],col==index_sample_node[j]]=j
        
        nodes_neighbours_one=_nodes_neighbours_one[_nodes_neighbours_info_one>=0,:]
        nodes_neighbours_one=torch.nn.functional.pad(nodes_neighbours_one,(0,3-dimension),value=0)
        nodes_neighbours_info_one=_nodes_neighbours_info_one[_nodes_neighbours_info_one>=0]

        
        y_one = ns_data['data_y'][index_sample_time, index_sample_node, :]

        # print(torch.mean(y_one),torch.std(y_one))
        # sns.kdeplot(y_one.reshape(-1),shade=True)
        # plt.show()
        


        nodes_state_batch.append(nodes_state_one.unsqueeze(0))
        nodes_neighbours_batch.append(nodes_neighbours_one.unsqueeze(0))
        nodes_neighbours_info_batch.append(nodes_neighbours_info_one.unsqueeze(0))

        



        y_batch.append(y_one.unsqueeze(0))

        f_token_batch.append(f_token)
        g_token_batch.append(g_token)
        
        f_equation_batch.append(eq_info.F_infix_c)
        g_equation_batch.append(eq_info.G_infix_c)
        f_ske_batch.append(eq_info.F_infix)
        g_ske_batch.append(eq_info.G_infix)
        dimension_batch.append(dimension)
    
    if len(f_token_batch) == 0:
        return None
        # return None, None, None, None, None, None, None, None, None,None,None
    else:
        
        nodes_state_batch=torch.cat(nodes_state_batch,dim=0)
        nodes_neighbours_batch=torch.cat(nodes_neighbours_batch,dim=0)
        nodes_neighbours_info_batch = torch.cat(nodes_neighbours_info_batch, dim=0)
        y_batch = torch.cat(y_batch, dim=0)


        
        # print(torch.mean(y_batch),torch.std(y_batch))
        # sns.displot(x=y_batch.reshape(-1),kind='kde')

        # pylab.show()

        
        
        f_token_batch = tokens_padding(f_token_batch)
        g_token_batch = tokens_padding(g_token_batch)


        data_info=DataInfo(
            Nodes_state=nodes_state_batch,
            Nodes_neighbours=nodes_neighbours_batch,
            Nodes_neighbours_info=nodes_neighbours_info_batch,
            Y=y_batch,
            F_token=f_token_batch,
            G_token=g_token_batch,
            F_equation=f_equation_batch,
            G_equation=g_equation_batch,
            F_ske=f_ske_batch,
            G_ske=g_ske_batch,
            Dimension=dimension_batch,
            Topo=topo,
            Index_sample_time=index_sample_time,
            Index_sample_node=index_sample_node,
            Origin_data=ns_data
        )

        return data_info
        #return nodes_state_batch, nodes_neighbours_batch, nodes_neighbours_info_batch, y_batch, f_token_batch, g_token_batch, f_equation_batch, g_equation_batch, f_ske_batch,g_ske_batch, dimension_batch
   

       
    

def custom_collate_fn(eqs_info, cfg, meta,mode):
    topo_nodes = np.random.randint(int(cfg.topo_max_nodes / 10), cfg.topo_max_nodes)
    topo_type = np.random.choice(a=cfg.topo_type_list, size=1, replace=None)
    topo = Topo(N=topo_nodes, topo_type=topo_type)
    topo_nodes = topo.N
    
    topo_nodes_have_neighbour=list(set(topo.sparse_adj[1].tolist()))
    
    num_sample_state=cfg.num_of_samples_state
    if cfg.few_sample_node:
        num_sample_nodes = 1
    else:
        num_sample_nodes = np.random.randint(1, min(int(len(topo_nodes_have_neighbour) / 2), 20))
        
    _index_sample_nodes = np.random.choice(a=topo_nodes_have_neighbour, size=num_sample_nodes, replace=False)
    num_steps_time = cfg.times
    index_sample_time = []
    index_sample_node = []
    num_sample_time_per_node=math.ceil(num_sample_state/num_sample_nodes)
    for ii in range(num_sample_nodes):
        if num_sample_time_per_node<num_steps_time:
            index_sample_time.append(torch.from_numpy(
                    np.random.choice(a=list(range(num_steps_time)), size=num_sample_time_per_node, replace=False)).long())
        else:
            index_sample_time.append(torch.from_numpy(
                    np.random.choice(a=list(range(num_steps_time)), size=num_sample_time_per_node, replace=True)).long())
        index_sample_node.append(torch.Tensor([_index_sample_nodes[ii]] * num_sample_time_per_node))
    
    index_sample_time = torch.cat(index_sample_time, dim=-1).long().view(-1)[:num_sample_state]
    index_sample_node = torch.cat(index_sample_node, dim=-1).long().view(-1)[:num_sample_state]
    print(f"sample: {num_sample_state}, sample node: {num_sample_nodes}")
    data_info=evaluate_and_wrap(eqs_info, topo, index_sample_time, index_sample_node, cfg, meta,mode)

    #node_state, neighbour_state, info, y, F_token, G_token, F_equations, G_equations, F_skeletons,G_skeletons, dimension=evaluate_and_wrap(eqs_info, topo, index_sample_time, index_sample_node, cfg, meta,mode)
    if data_info is None:
        print("get data Failure")
        return None

    else:
        print("get data Successed")
        return data_info
        #return node_state, neighbour_state, info, y, F_token, G_token, F_equations, G_equations, F_skeletons,G_skeletons, dimension,topo







def evaluate_and_wrap_only_F(eqs_info, index_sample_time, index_sample_node, cfg, meta,mode):
    
    nodes_state_batch = []
    
    y_batch = []
    f_token_batch = []
    
    f_equation_batch = []
    
    f_ske_batch=[]
    
    dimension_batch = []

    real_eqs_info=[]
    


    for i in range(len(eqs_info)):
        for j in range(cfg.num_of_samples_eq_test):
            f_infix_c=generate_float(str(eqs_info[i].F_infix),cfg.constant_range)
            real_eqs_info.append(Equation(F_prefix=eqs_info[i].F_prefix,
                                          G_prefix=eqs_info[i].G_prefix,
                                          F_infix=eqs_info[i].F_infix,
                                          G_infix=eqs_info[i].G_infix,
                                          F_infix_c=f_infix_c,
                                          G_infix_c="#####",
                                          dimension=eqs_info[i].dimension,
                                          id=eqs_info[i].id+"_"+str(j)))
    
    random.shuffle(real_eqs_info)
    for eq_info in real_eqs_info:
        try:
            f_token=tokenize(str(eq_info.F_prefix), meta["word2id"])
        except ValueError:
            continue
        except KeyError:
            continue
        
        dimension= eq_info.dimension
        ns_data=generate_datapoints_only_F(eq_info.F_infix_c,dimension,cfg.times)

        if ns_data is None:
            continue
    
        nodes_state_one = ns_data['data_node_state'][index_sample_time, index_sample_node, :]
        nodes_state_one=torch.nn.functional.pad(nodes_state_one,(0,3-dimension),value=0)
        
        
        y_one = ns_data['data_y'][index_sample_time, index_sample_node, :]

        


        nodes_state_batch.append(nodes_state_one.unsqueeze(0))

        y_batch.append(y_one.unsqueeze(0))

        f_token_batch.append(f_token)
        
        
        f_equation_batch.append(eq_info.F_infix_c)
        
        f_ske_batch.append(eq_info.F_infix)
  
        dimension_batch.append(dimension)
    
    if len(f_token_batch) == 0:
        return None
        # return None, None, None, None, None, None, None, None, None,None,None
    else:
        
        nodes_state_batch=torch.cat(nodes_state_batch,dim=0)
        y_batch = torch.cat(y_batch, dim=0)

        f_token_batch = tokens_padding(f_token_batch)




        data_info=DataInfo(
            Nodes_state=nodes_state_batch,
            Nodes_neighbours=None,
            Nodes_neighbours_info=None,
            Y=y_batch,
            F_token=f_token_batch,
            G_token=[],
            F_equation=f_equation_batch,
            G_equation=[],
            F_ske=f_ske_batch,
            G_ske=[],
            Dimension=dimension_batch,
            Topo=None,
            Index_sample_time=index_sample_time,
            Index_sample_node=index_sample_node,
            Origin_data=ns_data
        )

        return data_info











def custom_collate_fn_only_F(eqs_info, cfg, meta,mode):

    num_sample_state=cfg.num_of_samples_state
    num_sample_nodes = 1
 
    _index_sample_nodes = [0]
    num_steps_time = cfg.times
    index_sample_time = []
    index_sample_node = []
    num_sample_time_per_node=math.ceil(num_sample_state/num_sample_nodes)
    for ii in range(num_sample_nodes):
        if num_sample_time_per_node<num_steps_time:
            index_sample_time.append(torch.from_numpy(
                    np.random.choice(a=list(range(num_steps_time)), size=num_sample_time_per_node, replace=False)).long())
        else:
            index_sample_time.append(torch.from_numpy(
                    np.random.choice(a=list(range(num_steps_time)), size=num_sample_time_per_node, replace=True)).long())
        index_sample_node.append(torch.Tensor([_index_sample_nodes[ii]] * num_sample_time_per_node))
    
    index_sample_time = torch.cat(index_sample_time, dim=-1).long().view(-1)[:num_sample_state]
    index_sample_node = torch.cat(index_sample_node, dim=-1).long().view(-1)[:num_sample_state]
    print(f"sample: {num_sample_state}, sample node: {num_sample_nodes}")
    data_info=evaluate_and_wrap_only_F(eqs_info, index_sample_time, index_sample_node, cfg, meta,mode)

    #node_state, neighbour_state, info, y, F_token, G_token, F_equations, G_equations, F_skeletons,G_skeletons, dimension=evaluate_and_wrap(eqs_info, topo, index_sample_time, index_sample_node, cfg, meta,mode)
    if data_info is None:
        print("get data Failure")
        return None

    else:
        print("get data Successed")
        return data_info











class DataModule(pl.LightningDataModule):
    def __init__(self,train_path,val_path,test_path,cfg,metadata):
        super().__init__()

        self.train_path=train_path
        self.val_path=val_path
        self.test_path=test_path
        self.cfg=cfg
        self.metadata=metadata
    
    def setup(self, stage=None):
        if self.train_path:
            self.training_dataset=EqDataset(dataset_path=self.train_path,
                                            eqs_per_csv=self.metadata["eqs_per_csv_train"],
                                            len=self.metadata["total_number_of_eqs_train"])
            print("training_dataset:", len(self.training_dataset))
        if self.val_path:
            self.validation_dataset=EqDataset(dataset_path=self.val_path,
                                            eqs_per_csv=self.metadata["eqs_per_csv_val"],
                                            len=self.metadata["total_number_of_eqs_val"])
            print("validation_dataset:", len(self.validation_dataset))
        if self.test_path:
            self.test_dataset=EqDataset(dataset_path=self.test_path,
                                        eqs_per_csv=self.metadata["eqs_per_csv_test"],
                                        len=self.metadata["total_number_of_eqs_test"])
            print("test_dataset:", len(self.test_dataset))
    

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg, meta=self.metadata,mode="train"),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

        return trainloader
    
    def val_dataloader(self):
        validloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg, meta=self.metadata,mode="val"),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

        return validloader
    
    def test_dataloader(self):
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg, meta=self.metadata,mode="test"),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=False,
        )

        return testloader
    
    def test_dataloader_only_F(self):
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(custom_collate_fn_only_F, cfg=self.cfg, meta=self.metadata,mode="test"),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=False,
        )

        return testloader