from src.data_preprocessing.preprocess import preprocess_data

train_dir = "data/train"
validate_dir = "data/validate"
test_dir = "data/test"

train_data, train_labels = preprocess_data(train_dir)
validate_data, validate_labels = preprocess_data(validate_dir)
test_data, test_labels = preprocess_data(test_dir)

if __name__ == '__main__':
    print('Preprocessing complete.')
