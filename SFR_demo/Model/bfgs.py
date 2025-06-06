
import numpy as np
from data_factory_test import de_tokenize
import sympy as sp

import time
import numexpr as ne
import torch
import scipy
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from collections import Counter


class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value




    

def bfgs_F(F_ww,node_state,y,dimension,cfg,env):
    y=y.squeeze()
    F_pred_str = F_ww[1:].tolist()
    F_raw=de_tokenize(F_pred_str,cfg.id2word)
    try:
        F_eq_test=env.generator.prefix_to_tree(F_raw).infix()
    except:
        return None,None,None,None
    
    count_C=0
    F_expr = F_eq_test
    
    for i in range(F_eq_test.count("CONSTANT")):
        F_expr = F_expr.replace("CONSTANT", f"C{count_C}",1)
        count_C+=1
    
    # print('Constructing BFGS loss...')
    diffs = []
    F_variables=env.generator.variables[1:4]
    
    for i in range(node_state.shape[1]):
        F_curr_expr = F_expr
        for idx,j in enumerate(F_variables):
            F_curr_expr=sp.sympify(F_curr_expr).subs(j,node_state[:,i,idx])
        expr=F_curr_expr      
        diff = expr - y[i]
        diffs.append(diff)
    loss = 0
    loss = (np.mean(np.square(diffs))) 


    F_loss = []
    consts_ = []
    funcs = []


    vals_list=[]


    symbols = {i: sp.Symbol(f'C{i}') for i in range(count_C)}
    for i in range(1):
        x0 = np.random.randn(len(symbols))
        s = list(symbols.values())
        fun_timed = TimedFun(fun=sp.lambdify(s,loss, modules=['numpy']), stop_after=1e9)
    
        if len(x0):
            basinhopping(fun_timed.fun, x0, niter=1000, T=1.0, stepsize=0.5, minimizer_kwargs=None,
                            take_step=None,
                            accept_test=None, callback=None, interval=50, disp=False, niter_success=10)
            consts_.append(fun_timed.x)
        else:
            consts_.append([])

        F_final=F_expr

        for i in range(len(s)):
            F_final = sp.sympify(F_final).replace(s[i],np.around(fun_timed.x[i],decimals=4))
        funcs.append([F_final])


        F_local_dict={}

        for d in range(dimension):
            F_local_dict["x_i_{}".format(d)] = node_state[:,:, d].reshape(-1,1).cpu()

        vals = torch.from_numpy(ne.evaluate(str(F_final), local_dict=F_local_dict)).reshape(-1)


        _vals=torch.masked_select(vals,mask=torch.isnan(vals)!=True).numpy()
        _y=torch.masked_select(y.cpu(),mask=torch.isnan(vals)!=True).numpy()


        final_loss = np.mean(np.square(_vals-_y))

        F_loss.append(final_loss)
        vals_list.append(_vals)

    
    try:
        k_best = np.nanargmin(F_loss)
    except ValueError as e:
        print("All-Nan slice encountered")
        k_best = 0


    return funcs[k_best],F_loss[k_best],_y,vals_list[k_best]

def bfgs_FG(F_ww,G_ww,node_state,neighbour_state,info,y,dimension,cfg,env):
    y=y.squeeze()
    F_pred_str = F_ww[1:].tolist()
    G_pred_str = G_ww[1:].tolist()

    F_raw=de_tokenize(F_pred_str,cfg.id2word)
    G_raw=de_tokenize(G_pred_str,cfg.id2word)

    try:
        F_eq_test=env.generator.prefix_to_tree(F_raw).infix()
        G_eq_test=env.generator.prefix_to_tree(G_raw).infix()
    except:
        return None,None,None,None
    
    count_C=0
    F_expr = F_eq_test
    G_expr = G_eq_test

    for i in range(F_expr.count("CONSTANT")):
        F_expr = F_expr.replace("CONSTANT", f"C{count_C}",1)
        count_C+=1
    
   
    for i in range(G_expr.count("CONSTANT")):
        G_expr = G_expr.replace("CONSTANT", f"C{count_C}",1)
        count_C+=1
    
    diffs = []
    F_variables=env.generator.variables[1:4]
    G_variables=env.generator.variables[4:7]
    for i in range(node_state.shape[1]):
        F_curr_expr = F_expr
        G_curr_expr = G_expr
        for idx,j in enumerate(F_variables):
            F_curr_expr=sp.sympify(F_curr_expr).subs(j,node_state[:,i,idx])
            G_curr_expr=sp.sympify(G_curr_expr).subs(j,node_state[:,i,idx])
        expr=F_curr_expr
        
        for k in range(info.shape[1]):
            if i == info[:,k]:
                G_curr_expr_neighbour=G_curr_expr
                for idx,jj in enumerate(G_variables):
                    G_curr_expr_neighbour=sp.sympify(G_curr_expr_neighbour).subs(jj,neighbour_state[:,k,idx])
                expr=expr+G_curr_expr_neighbour 

        diff = expr - y[i]
        diffs.append(diff)
    loss = 0
    loss = (np.mean(np.square(diffs))) 


    F_loss = []
    consts_ = []
    funcs = []

    vals_list=[]
  
    symbols = {i: sp.Symbol(f'C{i}') for i in range(count_C)}
    for i in range(1):
        x0 = np.random.randn(len(symbols))
        s = list(symbols.values())
        fun_timed = TimedFun(fun=sp.lambdify(s,loss, modules=['numpy']), stop_after=1e9)
    
        if len(x0):
            basinhopping(fun_timed.fun, x0, niter=1000, T=1.0, stepsize=0.5, minimizer_kwargs=None,
                            take_step=None,
                            accept_test=None, callback=None, interval=50, disp=False, niter_success=10)
            consts_.append(fun_timed.x)
        else:
            consts_.append([])

        F_final=F_expr

        G_final=G_expr
        for i in range(len(s)):
            F_final = sp.sympify(F_final).replace(s[i],fun_timed.x[i])
    
        for i in range(len(s)):
            G_final = sp.sympify(G_final).replace(s[i],fun_timed.x[i])


        funcs.append([F_final,G_final])


        F_local_dict={}
        G_local_dict={}
        for d in range(dimension):
            F_local_dict["x_i_{}".format(d)] = node_state[:,:, d].reshape(-1,1).cpu()
            G_local_dict["x_i_{}".format(d)] = node_state[:,info.squeeze().long(), d].reshape(-1,1).cpu()
            G_local_dict["x_j_{}".format(d)] = neighbour_state[:,:, d].reshape(-1,1).cpu()
        try:
            F_vals = ne.evaluate(str(F_final), local_dict=F_local_dict).reshape(1,-1,1)
            G_vals = ne.evaluate(str(G_final), local_dict=G_local_dict).reshape(1,-1,1)

            F_vals=torch.from_numpy(F_vals).to(torch.float32)
            G_vals=torch.from_numpy(G_vals).to(torch.float32)
            vals = torch.scatter_add(F_vals, 1, info.squeeze().long().reshape(1,-1,1).cpu(), G_vals).reshape(-1)
        except Exception as e:
            print(e)
            print(str(F_final))
            print(str(G_final))
            print(F_vals.shape,G_vals.shape)
            continue

        _vals=torch.masked_select(vals,mask=torch.isnan(vals)!=True).numpy()
        _y=torch.masked_select(y.cpu(),mask=torch.isnan(vals)!=True).numpy()


        final_loss = np.mean(np.square(_vals-_y))

        F_loss.append(final_loss)
        vals_list.append(_vals)
    try:
        k_best = np.nanargmin(F_loss)
    except ValueError as e:
        print("All-Nan slice encountered")
        k_best = 0


    return funcs[k_best],F_loss[k_best],_y,vals_list[k_best]

def bfgs_FG_real_isomorphism(F_ww,G_ww,node_state,neighbour_state,info,y,dimension,cfg,env,raw_topo_data,index_sample_node):

    y=y.squeeze()


    F_pred_str = F_ww[1:].tolist()
    G_pred_str = G_ww[1:].tolist()
    
    F_raw=de_tokenize(F_pred_str,cfg.id2word)
    G_raw=de_tokenize(G_pred_str,cfg.id2word)

    F_eq_test=env.generator.prefix_to_tree(F_raw).infix()
    G_eq_test=env.generator.prefix_to_tree(G_raw).infix()

    count_C=0
    F_expr = F_eq_test
    G_expr = G_eq_test

    for i in range(F_expr.count("CONSTANT")):
        F_expr = F_expr.replace("CONSTANT", f"C{count_C}",1)
        count_C+=1
    
   
    for i in range(G_expr.count("CONSTANT")):
        G_expr = G_expr.replace("CONSTANT", f"C{count_C}",1)
        count_C+=1
    
    diffs = []
    F_variables=env.generator.variables[1:4]
    G_variables=env.generator.variables[4:7]
    

    for i in range(node_state.shape[1]):
        F_curr_expr = F_expr
        G_curr_expr = G_expr
        for idx,j in enumerate(F_variables):
            F_curr_expr=sp.sympify(F_curr_expr).subs(j,node_state[:,i,idx])
            G_curr_expr=sp.sympify(G_curr_expr).subs(j,node_state[:,i,idx])
        expr=F_curr_expr
        count=0

        for k in range(info.shape[1]):
            
            if i == info[:,k]:
                G_curr_expr_neighbour=G_curr_expr
                for idx,jj in enumerate(G_variables):
                    G_curr_expr_neighbour=sp.sympify(G_curr_expr_neighbour).subs(jj,neighbour_state[:,k,idx])
                    
                temp_node_neighbour_index=np.nonzero(raw_topo_data[index_sample_node[i]])
                expr=expr+raw_topo_data[index_sample_node[i]][temp_node_neighbour_index[0][count]]*G_curr_expr_neighbour
                expr=expr+G_curr_expr_neighbour
                count=count+1

        diff = expr - y[i]
        diffs.append(diff)
    loss = 0
    loss = (np.mean(np.square(diffs)))
    consts_ = []
    funcs = []

    symbols = {i: sp.Symbol(f'C{i}') for i in range(count_C)}
    for i in range(1):
        start_time=time.time()
        x0 = np.random.randn(len(symbols))
        s = list(symbols.values())
        fun_timed = TimedFun(fun=sp.lambdify(s,loss, modules=['numpy']), stop_after=1e9)
    
        if len(x0):
            basinhopping(fun_timed.fun, x0, niter=1000, T=1.0, stepsize=0.5, minimizer_kwargs=None,
                            take_step=None,
                            accept_test=None, callback=None, interval=50, disp=False, niter_success=10)
            consts_.append(fun_timed.x)
        else:
            consts_.append([])
        F_final=F_expr
        G_final=G_expr

        for i in range(len(s)):
            F_final = sp.sympify(F_final).replace(s[i],fun_timed.x[i])
    
        for i in range(len(s)):
            G_final = sp.sympify(G_final).replace(s[i],fun_timed.x[i])
        funcs.append([F_final,G_final])
    return funcs

  



