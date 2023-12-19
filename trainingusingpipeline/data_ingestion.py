import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')


class DataIngestion:

    def __init__(self, data_path, target_column) -> None:
        self.data_path = data_path
        self.target_column = target_column
        self.get_data()

    def get_data(self):
        try:
            dataframe = pd.read_csv(self.data_path)
            
            # Check if the target column is present in the dataframe
            if self.target_column not in dataframe.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

            X = dataframe.drop(self.target_column, axis=1)
            y = dataframe[self.target_column]

            return X, y

        except Exception as e:
            # Print a user-friendly error message or log the error
            print(f"Error during data ingestion: {str(e)}")
            # You might want to raise the exception again for further handling
            raise e
