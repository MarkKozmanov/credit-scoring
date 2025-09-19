import os
import pandas as pd


class DataLoader:

    def __init__(self, path):
        self.path = path

    def read_train_data(self):
        files = [f for f in os.listdir(self.path) if f.endswith("train.csv")]

        if not files:
            raise FileNotFoundError("train data were not found")

        train_file = files[0]
        train = pd.read_csv(os.path.join(self.path, train_file))
        return train

    def read_test_data(self):
        files = [f for f in os.listdir(self.path) if f.endswith('test.csv')]

        if not files:
            raise FileNotFoundError("test data were not found")
        test_file = files[0]
        test = pd.read_csv(os.path.join(self.path, test_file))
        return test

    def read(self):
        train = self.read_train_data()
        test = self.read_test_data()
        return train, test

    def save_data(self, output_dir, train, test, train_filename = "train.csv", test_filename = "test.csv"):
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, train_filename)
        test_path = os.path.join(output_dir, test_filename)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        return train_path, test_path



if __name__ == "__main__":
    input_path = os.path.join(os.getcwd(), 'data', 'raw')

    loader = DataLoader(input_path)

    try:
        train, test = loader.read()

        output  = os.path.join(os.getcwd(), "data", "interim")
        loader.save_data(output, train, test)

    except FileNotFoundError as e:
        print(f"Error {e}")



