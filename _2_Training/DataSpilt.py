from sklearn.model_selection import train_test_split

class DataSplit:
    def __init__(self, data) -> None:
        self.data = data

    def split_data(self):
        data = self.data
        total_len = len(data)

        test_size = 50
        val_size = 52

        train = data[: total_len - val_size - test_size]
        val = data[total_len - val_size - test_size : total_len - test_size]
        test = data[total_len - test_size :]

        return train, val, test


    # def split_data(self):
    #     # Random Seed
    #     RS = 3623
    #     non_test, test = train_test_split(self.data, test_size=0.1, random_state=RS, shuffle=True)
    #     train, val = train_test_split(non_test, test_size=1/9, random_state=RS, shuffle=True)
    #     return train, val, test