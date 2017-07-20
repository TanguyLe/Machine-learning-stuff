from six.moves import cPickle as pickle
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge

from notMNIST.constants import image_size
from notMNIST.paths import notMNIST_pickle_file, models_pickle_file
from utils.model import modelStore

with open(notMNIST_pickle_file, 'rb') as f:
    pickle_object = pickle.load(f)

    regr1 = LogisticRegression()
    regr2 = LogisticRegression()
    trees = tree.DecisionTreeClassifier()
    bayes = BayesianRidge()
    randomForestClass = RandomForestClassifier()
    gradientBoost = GradientBoostingClassifier()

    training_dataset = pickle_object['training_dataset']
    test_dataset = pickle_object['test_dataset']
    training_labels = pickle_object['training_labels']
    test_labels = pickle_object['test_labels']


    def shape_dataset(dataset):
        return dataset.reshape(len(dataset), image_size * image_size)


    print("Shaping...")
    # Train the model using the training sets
    new_training_dataset = shape_dataset(training_dataset)
    new_test_dataset = shape_dataset(test_dataset)

    print("Training...")
    # regr1.fit(new_test_dataset, test_labels)
    # regr2.fit(new_training_dataset, training_labels)
    # bayes.fit(new_training_dataset, training_labels)
    trees.fit(new_training_dataset, training_labels)
    # gradientBoost.fit(new_training_dataset, training_labels)
    # randomForestClass.fit(new_training_dataset, training_labels)

    model_store = modelStore(models_pickle_file)
    # model_store.register_model(regr1, "nominst_logreg_mini")
    # model_store.register_model(regr2, "nominst_logreg_full")
    # model_store.register_model(bayes, "nominst_bayes_full")
    model_store.register_model(trees, "nominst_trees_full")

    # model_store.register_model(gradientBoost, "nominst_gradboost_full")
    # model_store.register_model(randomForestClass, "nominst_randforest_full")

    print("Saving Model(s)...")
    model_store.save()
