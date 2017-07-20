from six.moves import cPickle as pickle
import os

base_path = "C:\\Users\\leflo\\Documents\\projects\\machine_learning_stuff\\"
model_file = base_path + "model.pickle"


class modelStore:
    def __init__(self, file=os.getcwd() + "\\model.pickle"):
        self.file_path = file
        print(self.file_path)
        try:
            with open(file, 'rb') as f:
                self.models = pickle.load(f)
        except (FileNotFoundError, EOFError) as e:
            self.models = dict()
            if e is FileNotFoundError:
                open(file, 'wb').close()

    def register_model(self, model, name):
        self.models[name] = model

    def get_model(self, name):
        return self.models[name]

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.models, f, pickle.HIGHEST_PROTOCOL)
