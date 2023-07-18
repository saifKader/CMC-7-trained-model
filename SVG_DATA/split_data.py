import os
import shutil
import numpy as np

def split_data_into_sets():
    root_dir = '/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/data/train/'
    train_dir = '/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/data/train/'
    validation_dir = '/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/data/validate/'
    test_dir = '/Users/abdelkaderseifeddine/Documents/GitHub/CMC-7-trained-model/data/test/'

    classes = os.listdir(root_dir)

    np.random.seed(0)

    for class_name in classes:
        # Skip the .DS_Store file
        if class_name == '.DS_Store':
            continue

        os.makedirs(train_dir + class_name, exist_ok=True)
        os.makedirs(validation_dir + class_name, exist_ok=True)
        os.makedirs(test_dir + class_name, exist_ok=True)

        all_files = os.listdir(os.path.join(root_dir, class_name))

        np.random.shuffle(all_files)

        num_images = len(all_files)

        train_idx = int(num_images * 0.7)  # 70% for training
        validation_idx = train_idx + int(num_images * 0.15)  # 15% for validation (70% + 15% = 85%)

        train_files = all_files[:train_idx]
        validation_files = all_files[train_idx:validation_idx]
        test_files = all_files[validation_idx:]  # 15% for testing

        for name in train_files:
            shutil.move(os.path.join(root_dir, class_name, name), os.path.join(train_dir, class_name, name))
        for name in validation_files:
            shutil.move(os.path.join(root_dir, class_name, name), os.path.join(validation_dir, class_name, name))
        for name in test_files:
            shutil.move(os.path.join(root_dir, class_name, name), os.path.join(test_dir, class_name, name))

    print('Data split into training, validation, and test sets.')

split_data_into_sets()
