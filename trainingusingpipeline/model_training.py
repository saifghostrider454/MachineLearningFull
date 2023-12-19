from pipelines import pipeLines
from data_ingestion import DataIngestion


def train_model():
    data_ingestion = DataIngestion('hiring.csv', 'salary($)')
    X, y = data_ingestion.get_data()
    pipe = pipeLines()

    model = pipe.fit(X, y)

    return model
