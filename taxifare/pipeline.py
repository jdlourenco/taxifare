from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from taxifare.encoders import DistanceTransformer

def get_model(estimator="RandomForest"):
    if estimator == "RandomForest":
        model_params = dict(n_estimators=100, max_depth=1)

        model = RandomForestRegressor()
        model.set_params(**model_params)
    elif estimator == "LinearRegression":
        model = LinearRegression()
        
    return model

def get_pipeline(estimator="RandomForest"):
    pipe_distance = make_pipeline(
        DistanceTransformer(),
        StandardScaler())


    cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    feateng_blocks = [
        ('distance', pipe_distance, cols),
    ]

    features_encoder = ColumnTransformer(feateng_blocks)

    pipeline = Pipeline(
        steps=[
            ('features', features_encoder),
            ('model', get_model(estimator=estimator))
        ]
    )
    
    return pipeline
