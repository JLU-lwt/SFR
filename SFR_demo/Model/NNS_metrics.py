import numpy as np

def cal_metrics(true_data,pre_data):
    
    #close
    metrics_close1=np.isclose(pre_data,true_data,atol=0,rtol=1)
    metrics_close01=np.isclose(pre_data,true_data,atol=0,rtol=0.1)
    metrics_close001=np.isclose(pre_data,true_data,atol=0,rtol=0.01)
    metrics_close0001=np.isclose(pre_data,true_data,atol=0,rtol=0.001)

    metrics_close1=metrics_close1.mean()
    metrics_close01=metrics_close01.mean()
    metrics_close001=metrics_close001.mean()
    metrics_close0001=metrics_close0001.mean()



    # S_metrics_close01=float(metrics_close01.mean()>0.95)
    # S_metrics_close001=float(metrics_close001.mean()>0.95)
    # S_metrics_close0001=float(metrics_close0001.mean()>0.95)




    #R^2
    metric_R2=1-((np.sum(np.square(pre_data-true_data)))/(np.sum(np.square(np.mean(true_data)-true_data))))


    #mse

    metric_MSE=np.mean(np.square(true_data-pre_data))


    #mape

    metric_MAPE=np.mean(np.abs((true_data-pre_data)/true_data))



    output = {  
                'close_1':metrics_close1,
                'close_0.1':metrics_close01, 
                'close_0.01':metrics_close001, 
                'close_0.001':metrics_close0001, 
                # 'S_close_0.1':S_metrics_close01,
                # 'S_close_0.01':S_metrics_close001,
                # 'S_close_0.001':S_metrics_close0001,
                'R^2':metric_R2,
                'MSE':metric_MSE,
                'MAPE': metric_MAPE
                }
    
    return output