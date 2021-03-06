from taxifare.data import get_data, clean_data, holdout
from taxifare.pipeline import get_pipeline
from taxifare.utils import compute_rmse
import joblib
from google.cloud import storage
from taxifare.data import BUCKET_NAME, BUCKET_MODEL_PATH

class Trainer:
    def __init__(self, nrows=100, estimator="RandomForest"):
        self.pipeline = None
        self.X_train = None
        self.y_train = None
        self.nrows = nrows
        self.estimator = estimator

    def train_model(self):
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline
    
    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse
    
    def upload_to_gcp(self, filename):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{BUCKET_MODEL_PATH}/{filename}")
        blob.upload_from_filename(filename)
    
    def save_model(self):
        filename = f"{self.estimator}.joblib"
        joblib.dump(self.pipeline, filename)
        self.upload_to_gcp(filename)
    
    def train(self):
        print("get data")
        df = get_data(nrows=self.nrows)
        print(df.shape)
        
        print("clean data")
        df_clean = clean_data(df)
        print(df_clean.shape)
        
        print("split into X and y")
        print("holdout")
        (X_train, X_test, y_train, y_test) = holdout(df_clean)
        self.X_train = X_train
        self.y_train = y_train
        
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        
        print("define model: create a pipeline")
        self.pipeline = get_pipeline(estimator=self.estimator)
        print(self.pipeline)

        print("train model")
        self.train_model()
        print("evaluate")
        rmse = self.evaluate(X_test, y_test)
        print(f"rmse={rmse}")

        print("save model")
        self.save_model()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
