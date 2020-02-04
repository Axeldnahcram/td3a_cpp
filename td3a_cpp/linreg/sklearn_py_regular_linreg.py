# function file
# sklearn_regular_linreg - sklearn LinearRegression

from sklearn.linear_model import ElasticNet

def sklearn_regular_linreg(X, y, alpha, L1_ratio, *args, **kwargs):
    """
    sklearn regularized linear regression 

    :return: slope of the regression
    """

    X = X.astype(numpy.float64)
    y = y.astype(numpy.float64)
    reg = ElasticNet(alpha=alpha, l1_ratio=L1_ratio)
    reg.fit(X,y)

    return reg.coef_
