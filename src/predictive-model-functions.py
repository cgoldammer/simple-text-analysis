'''
Created on Apr 1, 2014

@author: cg
'''

def get_cutoffs(x,num_groups=10):
    series=Series(x)
    cutoffs=[]
    for i in range(num_groups):
        perc_low=float(i)/num_groups
        perc_high=float(i+1)/num_groups
        cutoffs.append((series.quantile(perc_low),series.quantile(perc_high)))
    return cutoffs

def share_correct(y,y_hat,verbose=False):
    """This function is only relevant for binary models. For these models, it shows the percentage of predictions
    correctly classified using the prediction y_hat. This assumes that we classify y=1 if y_hat>.5 and y=0 otherwise"""
    df=pd.DataFrame({"y":y,"y_hat":y_hat})
    df["y_classifier"]=(df.y_hat>.5)
    df["correctly_classified"]=df.y_classifier==df.y
    return df.correctly_classified.mean()

def mean_outcome_in_groups(y,y_hat,num_groups=10,verbose=False):
    """Get the average of the outcome y when y_hat is cut into num_groups equally-sized groups. This 
    is used as a measure of performance of the model"""
    cutoffs=get_cutoffs(y_hat,num_groups)
    return mean_outcome_by_cutoff(y,y_hat,cutoffs)

def mean_outcome_by_cutoff(y,y_hat,cutoffs,verbose=False):
    """Show the average outcome y by the cutoffs for y_hat"""
    y_by_group=[]
    df=pd.DataFrame({"y":y,"y_hat":y_hat})
    # Get performance from test sample (test==2), not from valdiation sample (test==1)
    for cutoff_low,cutoff_high in cutoffs:
        data_group=df[(df.y_hat>=cutoff_low) & (df.y_hat<cutoff_high)]  
        y_by_group.append(np.mean(data_group["y"]))
    performance=[]
    for i in range(len(cutoffs)):
        performance.append((i+1,round(y_by_group[i],3)))
    return performance

def convert_performance_to_string(performance):
    performance_string=[]
    note=""
    for decile in performance:
        value=decile[1]
        value_string=str(value)
        if math.isnan(value):
            value_string="N/A*"
            note="Since deciles are determined using the training sample, it cannot be ensured that all deciles can be evaluated in the test sample"
        decile_string=(decile[0],value_string)
        performance_string.append(decile_string)
    return (performance_string,note)