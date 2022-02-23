import pandas as pd
import joblib
from google.cloud import storage
from taxifare.data import BUCKET_NAME

TEST_PATH = "raw_data/test.csv"
# MODEL_PATH = "RandomForest.joblib"
MODEL_PATH = "gs://wagon-data-804-jdlourenco/wagon-804/taxifare/models/RandomForest.joblib"

# get test data
def get_test_data():
    return pd.read_csv(TEST_PATH)

# get model
def get_model(model_path):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("wagon-804/taxifare/models/RandomForest.joblib")
    blob.download_to_filename("model.joblib")
    return joblib.load("model.joblib")

if __name__ == "__main__":
    test_set = get_test_data()
    print(test_set.head())

    model = get_model(MODEL_PATH)
    print(model)

    pred = model.predict(test_set)
    print(pred)

    test_set["fare_amount"] = pred
    test_set = test_set[["key", "fare_amount"]]
    test_set.to_csv("results.csv", index=False)

