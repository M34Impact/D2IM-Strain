from sklearn.model_selection import train_test_split

class DataSplit:
    def __init__(self, data) -> None:
        self.data = data

    def split_data(self):
        # Random Seed
        RS = 3623
        non_test, test = train_test_split(self.data, test_size=0.1, random_state=RS, shuffle=True)
        train, val = train_test_split(non_test, test_size=1/9, random_state=RS, shuffle=True)
        return train, val, test