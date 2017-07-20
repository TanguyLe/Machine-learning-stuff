from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

from utils.model import modelStore
from utils.image import PIL_Image

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
base_path = "C:\\Users\\leflo\\Documents\\projects\\machine_learning_stuff\\"
pickle_file = base_path + "data\\notMNIST\\notMNIST.pickle"
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

with open(pickle_file, 'rb') as f:
    pickle_object = pickle.load(f)

    # Create linear regression object
    regr = LogisticRegression()
    training_dataset = pickle_object['train_dataset']
    test_dataset = pickle_object['test_dataset']
    training_labels = pickle_object['train_labels']
    test_labels = pickle_object['test_labels']


    def shape_dataset(dataset):
        return dataset.reshape(len(dataset), image_size * image_size)


    def unnormalize(image):
        return (image * pixel_depth) + pixel_depth / 2


    print("Shaping...")
    # Train the model using the training sets
    new_training_dataset = shape_dataset(training_dataset)
    new_test_dataset = shape_dataset(test_dataset)

    new_model_store = modelStore()
    classif1 = new_model_store.get_model("regr_test")

    print("Score Mini test: " + str(classif1.score(new_test_dataset, test_labels)))
    print("Score Mini train: " + str(classif1.score(new_training_dataset, training_labels)))

    classif2 = new_model_store.get_model("nominst_logreg_full")

    print("Score Full test: " + str(classif2.score(new_test_dataset, test_labels)))
    print("Score Full train: " + str(classif2.score(new_training_dataset, training_labels)))

    classif3 = new_model_store.get_model("nominst_bayes_full")
    classif4 = new_model_store.get_model("nominst_trees_full")
    classif5 = new_model_store.get_model("nominst_randforest_full")
    classif6 = new_model_store.get_model("nominst_gradboost_full")

    print("Score Bayes test: " + str(classif3.score(new_test_dataset, test_labels)))
    print("Score Bayes train: " + str(classif3.score(new_training_dataset, training_labels)))

    print("Score Trees test: " + str(classif4.score(new_test_dataset, test_labels)))
    print("Score Trees train: " + str(classif4.score(new_training_dataset, training_labels)))

    print("Score RandomForest test: " + str(classif5.score(new_test_dataset, test_labels)))
    print("Score RandomForest train: " + str(classif5.score(new_training_dataset, training_labels)))

    # print("Score GradBoost test: " + str(classif6.score(new_test_dataset, test_labels)))
    # print("Score GradBoost train: " + str(classif6.score(new_training_dataset, training_labels)))

    # pred1 = classif1.predict(new_test_dataset)
    # pred2 = classif2.predict(new_test_dataset)
    # pred3 = classif3.predict(new_test_dataset)
    # pred4 = classif4.predict(new_test_dataset)
    # pred5 = classif5.predict(new_test_dataset)
    # pred6 = classif6.predict(new_test_dataset)
    #
    # for i in range(10):
    #         print("\nImage " + str(i) + ':' + letters[test_labels[i]])
    #         img = PIL_Image(array=unnormalize(new_test_dataset[i].reshape(28, 28)))
    #         img.show()
    #         print("Mini:" + str(letters[pred1[i]]) + " / Full:" + str(letters[pred2[i]]))
    #         print("Bayes:" + str(letters[int(round(pred3[i]))]) + " / Trees:" + str(letters[pred4[i]]))
    #         print("RandomForest:" + str(letters[pred5[i]]) + " / GradBoost:" + str(letters[pred6[i]]))
