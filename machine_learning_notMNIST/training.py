from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from six.moves import cPickle as pickle

from utils.model import modelStore

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
base_path = "C:\\Users\\leflo\\Documents\\projects\\machine_learning_stuff\\"
pickle_file = base_path + "data\\notMNIST\\notMNIST.pickle"

with open(pickle_file, 'rb') as f:
    pickle_object = pickle.load(f)

    # BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    # RandomForestClassifier(n_estimators=10)
    # linear_model.BayesianRidge()
    # KNeighborsClassifier
    # GaussianProcessClassifier
    # tree.DecisionTreeClassifier()
    # Create linear regression object
    regr1 = LogisticRegression()
    regr2 = LogisticRegression()
    trees = tree.DecisionTreeClassifier()
    bayes = BayesianRidge()
    randomForestClass = RandomForestClassifier()
    gradientBoost = GradientBoostingClassifier()

    training_dataset = pickle_object['train_dataset']
    test_dataset = pickle_object['test_dataset']
    training_labels = pickle_object['train_labels']
    test_labels = pickle_object['test_labels']


    def normalize_and_shape_dataset(dataset):
        return dataset.reshape(len(dataset), image_size * image_size)


    print("Shaping...")
    # Train the model using the training sets
    new_training_dataset = normalize_and_shape_dataset(training_dataset)
    new_test_dataset = normalize_and_shape_dataset(test_dataset)
    print("Training...")
    # regr1.fit(new_test_dataset, test_labels)
    # regr2.fit(new_training_dataset, training_labels)
    # bayes.fit(new_training_dataset, training_labels)
    # trees.fit(new_training_dataset, training_labels)
    gradientBoost.fit(new_training_dataset, training_labels)
    randomForestClass.fit(new_training_dataset, training_labels)

    model_store = modelStore()
    # model_store.register_model(regr1, "nominst_logreg_mini")
    # model_store.register_model(regr2, "nominst_logreg_full")
    # model_store.register_model(bayes, "nominst_bayes_full")
    # model_store.register_model(trees, "nominst_trees_full")

    model_store.register_model(gradientBoost, "nominst_gradboost_full")
    model_store.register_model(randomForestClass, "nominst_randforest_full")

    model_store.save()
