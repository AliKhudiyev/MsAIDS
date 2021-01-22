import numpy as np
from sklearn.metrics import classification_report

'''
    Value-Position Interpolation
    ----------------------------

    VPI is a simple model which is very easy to understand. All it does is to predict the value(category)
    of a given data point by using a simple interpolation according to the euclidean distances to the points 
    which are closer to it. The number of points to be chosed is given by the user and the optimal value
    for the number of closest points can be verified by iterating through the given domain of n(number of 
    closest points).

    The interpolation has the formula shown below:

                            c = \sum_{i=1}^{n} c / (||p_i p|| + 1)
                    where c is the category of the point p and p_i is one the many closest points to p
    
    To normalize the obtained value, it is multiplied by the normalizing coeffcient:

                            c_norm = c / [ \sum_{i=1}^{n} 1 / (||p_i p|| + 1) ]
                    where ||p_i p|| is the euclidean distance between the points p_i and p
    
    The final result or category of the point p is given by c_norm. So, the whole thing works by applying
    ratios of distances to their categories. Unfortunately, this model lacks the learning process which means
    there is no way for the model to learn from its mistakes and perform better in the next time. Another 
    disadvantage would be the computation time since each time we want to predict the category/label of a 
    data point, it should sort all the known data points according to their distances relative to the given 
    point. In the other hand, the advantage of this method is the fact that it works reasonably better when
    the proximity of two data points is proportional to the similarity of their categories.
'''

def vpi(X, X_known, y_known, n=1, dist_func=None):
    y = list()
    
    for X_ in X:
        dists = [np.linalg.norm(np.array([x1 - x2 for x1, x2 in zip(X_, X2)])) for X2 in X_known]
        indices = list(range(len(y_known)))
        dists_sorted, indices_sorted = zip(*sorted(zip(dists, indices)))
        dists_sorted, indices_sorted = (list(t) for t in zip(*sorted(zip(dists_sorted, indices_sorted))))

        y_ = 0
        norm_coef = 0
        for i in range(n):
            if i >= len(y_known):
                break
            
            if dist_func is not None:
                pass
            else:
                coef = (np.linalg.norm(np.array([x1 - x2 for x1, x2 in zip(X_, X_known[indices_sorted[i]])])) + 1)
                y_ += y_known[indices_sorted[i]] / coef
                norm_coef += 1 / coef
        y.append(y_ / norm_coef)
    
    return y
