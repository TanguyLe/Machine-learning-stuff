import os


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))

base_path = get_parent_dir(__file__)
data_path = os.path.join("data")
machine_learning_path = os.path.join("machine_learning")

notMNIST_small_path = os.path.join(data_path, "notMNIST_small")
notMNIST_large_path = os.path.join(data_path, "notMNIST_large")

test_base_path = os.path.join(base_path, notMNIST_small_path)
training_base_path = os.path.join(base_path, notMNIST_large_path)

notMNIST_pickle_file = os.path.join(base_path, data_path, "notMNIST.pickle")

models_pickle_file = os.path.join(base_path, machine_learning_path, "model.pickle")
