from sklearn.model_selection import train_test_split

class DataSplit:
    def __init__(self, data) -> None:
        self.data = data

    def split_data(self):
        # Random Seed
        RS = 3623
        target_ezz_NT, target_ezz_test = train_test_split(self.data, test_size=0.1, random_state=RS, shuffle=True)
        target_ezz_train, target_ezz_val = train_test_split(target_ezz_NT, test_size=1/9, random_state=RS, shuffle=True)
        return target_ezz_train, target_ezz_val, target_ezz_test