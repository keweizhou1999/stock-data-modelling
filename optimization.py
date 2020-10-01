import cluster2 as cluster
from sklearn import metrics
from scipy.optimize import minimize



def opt_helper(k, weight_, raw_data):
    """
    Calculate silhouette score based on given weight and k
    """
    # normalize the raw data so that they are all in the range of (0,1)
    (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)
    # define the number of clusters 
    #weight_ = [weight[0], weight[1], weight[2], 0]
    print("weight:",weight_)
    
    def distance(item1, item2):
        return cluster.distance(weight_,item1, item2)

    clustering = cluster.cluster(weight_, norm_data, k)
    # clusters = cluster.display(norm_data, clustering, k)

    result = -metrics.silhouette_score(norm_data, clustering, 
            metric = distance, sample_size=1000, random_state=2)
    print("result: ", result)
    return result


def minimizeHelper(rosen, weight0):
    """
    Different scipy methods to optimize
    Non-linear constraints and bounds included
    """
    # constraint for SLSQP and COBYLA format
    constr = {'type':'ineq',
            'fun': lambda x:1-sum(x)
            }

    # constraint for trust-constr method
    # constraint = NonlinearConstraint(sum, 1, 1)

    #bounds for the variables
    bounds = tuple(((0,1) for x in weight0))

    
    # Different methods tried for optimization

    # ans = minimize(rosen, weight0, method='Nelder-Mead', 
    #             options={'maxiter':1})
    ans =  minimize(rosen, weight0, method='SLSQP', 
                    constraints=[constr],bounds=bounds,
                    options={'maxiter':2})
    # ans = minimize(rosen, weight0, method='COBYLA', 
    #                 constraints=[constr],
    #                 options={'tol':0.1,'maxfev':2})
    # ans = minimize(rosen, weight0, method='trust-constr', 
    #                 bounds=bounds,constraints=[constraint], 
    #                 options={'maxiter':3})
    print(ans)
    return ans

def manual_minimize(rosen):
    """
    Manually try different weights to get Silhouette scores
    """
    testing_weight = [[0, 0.5, 0.5, 0],
                    [0.15, 0.75, 0.05, 0.05],
                    [0.155, 0.745, 0.05, 0.05],
                    [0.14, 0.76, 0.05, 0.05],
                    [0.14, 0.74, 0.15, 0.05],
                    [0.2, 0.78, 0.015, 0.005],
                    [1, 0, 0, 0],
                    [0, 1, 0 , 0],
                    [0.1, 0.8, 0.1, 0],
                    [0.15, 0.75, 0.1, 0],
                    [0.2, 0.75, 0.05, 0],
                    [0.15, 0.75, 0.05, 0.05],
                    [0.3, 0.3, 0.3, 0.1],
                    [0.2, 0.5, 0.2, 0.1],
                    [0.2, 0.2, 0.6, 0],
                    [0.13, 0.22, 0.65, 0]]

    for weight_ in testing_weight:
        score = rosen(weight_)
        print("weight, score:", weight_, score)


