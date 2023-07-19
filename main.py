from src.data_preprocessing.preprocess import preprocess_data
from src.model_training.train import create_model, train_model

train_dir = "data/train"
validate_dir = "data/validate"
test_dir = "data/test"

train_data, train_labels = preprocess_data(train_dir)
validate_data, validate_labels = preprocess_data(validate_dir)
test_data, test_labels = preprocess_data(test_dir)

input_shape = train_data.shape[1:]  # assuming all images have the same shape
num_classes = train_labels.shape[1]  # number of classes is number of unique labels
epochs = 50  # define how many epochs to train for
model_path = "models/cnn_model/model.tf"  # where to save the model

if __name__ == '__main__':
    print('Preprocessing complete.')
    model = create_model(input_shape, num_classes)
    train_model(model, train_data, train_labels, validate_data, validate_labels, epochs, model_path)
    print('Model training complete.')
