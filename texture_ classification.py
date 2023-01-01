import cv2
import numpy as np
import os
import glob
import mahotas as mt

# load the training dataset
train_path = "dataset/train"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = []
train_labels = []

def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
    
# loop over the training dataset
print ("[STATUS] Started extracting haralick textures..")
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        i = 1
        for file in glob.glob(cur_path + "/*.png"):
                # read the training image
                image = cv2.imread(file)

                # convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # extract haralick texture from the image
                features = extract_features(gray)

                # append the feature vector and label
                train_features.append(features)
                train_labels.append(cur_label)

                # show loop update
                i += 1

# create the classifier
print ("[STATUS] Creating the classifier..")

#from sklearn.svm import LinearSVC
#clf = LinearSVC(random_state=9, max_iter = 200000)

from sklearn.tree import DecisionTreeClassifier

#from sklearn.naive_bayes import GaussianNB

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.ensemble import RandomForestClassifier

# fit the training data and labels
print ("[STATUS] Fitting data/label to model..")
#clf.fit(train_features, train_labels)

clf = DecisionTreeClassifier(max_depth=3).fit(train_features,train_labels)

#clf = GaussianNB().fit(train_features, train_labels)

#clf = KNeighborsClassifier(n_neighbors = 5).fit(train_features, train_labels)

#clf = RandomForestClassifier(max_depth=3).fit(train_features, train_labels)

# loop over the test images
test_path = "dataset/test"
for file in glob.glob(test_path + "/*.png"):
	# read the input image
	image = cv2.imread(file)

	# convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# extract haralick texture from the image
	features = extract_features(gray)

	# evaluate the model and predict label
	prediction = clf.predict(features.reshape(1, -1))[0]

	# show the label
	cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	#print ("Prediction - {}").format(prediction)

	# display the output image
	cv2.imshow("Test_Image", image)
	cv2.waitKey(0)
