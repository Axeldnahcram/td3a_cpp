# function file
# sklearn_regular_linreg - sklearn LinearRegression

from sklearn.linear_model import LinearRegression

def sklearn_regular_linreg(X, y, beta_0, alpha, L1_ratio, max_iter=1000, tol=0.0001, *args, **kwargs):
    """
    sklearn regular linear regression 

    :return: slope of the regression
    """

    reg = LinearRegression().fit(X, y)

    return reg.coef_
