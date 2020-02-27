from sklearn.linear_model import ElasticNet

def get_ElasticNet(alpha, l1_ratio=0.5):
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio)