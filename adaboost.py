import numpy as np
import random as rand
import pickle


class WeakClassifier:
    
    '''
    feature_index -> the feature being used in this classifier
    threshold -> the cutoff between feature and non_feature
    polarity -> 1 indicating if right of threshold is match or -1 for non match
    '''
    def __init__ (self, feature_index = None, threshold = None, polarity = None, alpha = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.polarity = polarity
        self.alpha = alpha

    '''
    im_featuers -> list of image features
    '''
    def classify (self, im_features):
        # print("Feature_Index " + str(self.feature_index))
        # print("threshold " + str(self.threshold))
        # print("polarity " + str(self.polarity))
        # print("alpha " + str(self.alpha))

        if im_features[self.feature_index] < self.threshold:
            return -1 * self.polarity
        return self.polarity

    '''
    im_feature_matrix -> matrix where rows are images and cols are features applied on the images
    labels -> list of the labels showing correct classification
    weights -> the weights of each image
    '''
    def train (self, im_features_matrix, labels, weights):

        min_error = float("inf")

        for feature in range(np.size(im_features_matrix, 1)):

            feature_column = []
            for index, f in enumerate(im_features_matrix[:, feature]):
                feature_column.append([f, index])

            feature_column.sort(key=lambda a: a[0])

            predictions = np.ones(np.size(im_features_matrix, 0))
            predictions[im_features_matrix[:, feature] < feature_column[0][0]] = -1

            #error if everything to the right is predicted a match and everyting to the left is predicted not match
            error = sum([weight for index, weight in enumerate(weights) if predictions[index] != labels[index]])

            for i in range(1, len(feature_column)):

                if labels[feature_column[i - 1][1]] == -1:
                    error -= weights[feature_column[i - 1][1]]
                else:
                    error += weights[feature_column[i - 1][1]]

                temp_error = error
                p = 1

                if temp_error > 0.5:
                    p = -1
                    temp_error = 1 - temp_error

                if temp_error < min_error:
                    min_error = temp_error
                    self.polarity = p
                    self.threshold = feature_column[i][0]
                    self.feature_index = feature


        NON_ZERO_DENOM = 0.0000001
        self.alpha = 0.5 * np.log((1.0 - min_error + NON_ZERO_DENOM) / (min_error + NON_ZERO_DENOM))
        print("ERROR " + str(min_error))

    def package (self):
        return {
            "feature_index" : self.feature_index,
            "threshold" : self.threshold,
            "polarity" : self.polarity,
            "alpha" : self.alpha
        }



class ImAdaBoost:

    '''
    num_features -> number of features in this model
    num_classifiers -> debatably should be in train
    im_size -> the size of the input images
    '''
    def __init__ (self, num_features, im_size):
        self.features = [ImFeature(im_size) for _ in range(num_features)]
        self.im_size = im_size
        self.weak_classifiers = []
        self.weights = []
    
    '''
    train_images -> list of Image type training images
    labels -> list of labels true -> match false -> not match
    '''
    def train (self, train_images, labels, number_of_classifiers):
        weights = self.weights

        for weight_index in range(len(labels)):
            if labels[weight_index] == -1:
                weights.append(0.5 / (0.9 * len(labels)))
            else:
                weights.append(0.5 / (0.1 * len(labels)))

        im_features_matrix = np.zeros((len(train_images), len(self.features)))

        for im_index, im in enumerate(train_images):
            for i in range(len(self.features)):
                im_features_matrix[im_index, i] = self.features[i].apply_on_image(im)
            if ((im_index + 1) % 5000 == 0):
                print("Described " + str(im_index + 1) + " images as features")
        
        for current_classifier in range(number_of_classifiers):
            weak_classifier = WeakClassifier()
            weak_classifier.train(im_features_matrix, labels, weights)

            for im_index in range(np.size(train_images, 0)):
                weights[im_index] *= np.exp(-weak_classifier.alpha * labels[im_index] * weak_classifier.classify(im_features_matrix[im_index, :]))
            
            weights /= sum(weights)

            self.weak_classifiers.append(weak_classifier)   

            print("Trained " + str(current_classifier + 1) + " weak classifiers")
       
        self.weights = weights    
    
    def predict (self, im):
        features = [feature.apply_on_image(im) for feature in self.features]
        return self.predict_on_features(features)

    def predict_on_features (self, features):
        predictions = [weak_classifier.classify(features) * weak_classifier.alpha for weak_classifier in self.weak_classifiers]
        return np.sum(predictions)
    
    def store_model (self, file_name):
        features = [feature.package() for feature in self.features]
        classifiers = [clsf.package() for clsf in self.weak_classifiers]
        data = {
            "features" : features,
            "classifiers" : classifiers,
            "weights" : self.weights,
            "im_size" : self.im_size
        }

        file = open(file_name, "wb")
        pickle.dump(data, file)
        file.close()
    
    def reconstruct_model (self, file_name):
        file = open(file_name, "rb")
        data = pickle.load(file)
        self.weights = data["weights"]
        self.im_size = data["im_size"]

        self.features = []
        for feature in data["features"]:
            self.features.append(ImFeature(feature["im_size"], feature["type"], feature["corners"], feature["borders"]))
 
        self.weak_classifiers = []
        for classifier in data["classifiers"]:
            self.weak_classifiers.append(WeakClassifier(classifier["feature_index"], classifier["threshold"], classifier["polarity"], classifier["alpha"]))




class Image:

    '''
    im -> image as a numpy array 
    '''
    def __init__ (self, im):

        self.im = im
        self.im_size = [np.size(im, 0), np.size(im, 1)] 
        self.calc_integral_image()

    '''
    calcualtes the integral image of an image and stores it in self.integral_image
    '''  
    def calc_integral_image (self):

        self.integral_image = np.zeros((self.im_size[0], self.im_size[1]), dtype=int)

        self.integral_image[0, 0] = self.im[0, 0]

        for i in range(1, self.im_size[0]):
            self.integral_image[0, i] = self.im[0, i] + self.integral_image[0, i - 1]
        
        for i in range(1, self.im_size[1]):
            self.integral_image[i, 0] = self.im[i, 0] + self.integral_image[i - 1, 0]

        for i in range(1, self.im_size[0]):
            for j in range(1, self.im_size[1]):
                self.integral_image[i, j] = self.im[i, j] + self.integral_image[i - 1, j] + self.integral_image[i, j - 1] - self.integral_image[i - 1, j - 1]

    '''
    cornerA -> top left corner of the integral image
    cornerB -> bottom left corner of the integral image
    '''
    def get_rect (self, corner_A, corner_B):
        return self.integral_image[corner_A[0], corner_A[1]] - \
            self.integral_image[corner_A[0], corner_B[1]] - \
            self.integral_image[corner_B[0], corner_A[1]] + \
            self.integral_image[corner_B[0], corner_B[1]]


class ImFeature:

    # types of features
    # 0 -> 2-Rectangle
    # 1 -> 3-Rectangle
    # 2 -> 4-Checkerboard

    '''
    im_size -> the size of the image the feature is being created for
    paramaters for reconstruction of model
    '''
    def __init__ (self, im_size, type = None, corners = None, borders = None):
        self.typesOfFeatures = 3
        self.im_size = im_size
        if corners == None or borders == None or type == None:
            self.generate_feature ()
        else:
            self.corners = corners
            self.borders = borders
            self.type = type
        

    '''
    randomly generates a harr feature
    '''
    def generate_feature(self):
        # self.type = rand.randint(0, self.typesOfFeatures - 1)
        self.type = 0

        if self.type == 0:
            self.generate_2_rectangle ()
        elif self.type == 1:
            self.generate_3_rectangle ()
        elif self.type == 2:
            self.generate_4_checkerboard ()

    '''
    generation of a 2 rectangle type feature
    '''
    def generate_2_rectangle (self):
        MIN_RECTANGLE_WIDTH = 5
        row1 = rand.randint(0, self.im_size[0] - MIN_RECTANGLE_WIDTH)
        row2 = rand.randint(row1 + MIN_RECTANGLE_WIDTH - 1, self.im_size[0] - 1)

        col1 = rand.randint(0, self.im_size[1] - MIN_RECTANGLE_WIDTH)
        col2 = rand.randint(col1 + MIN_RECTANGLE_WIDTH - 1, self.im_size[1] - 1)

        self.corners = [[row1, col1], [row2, col2]]
        self.borders = [rand.randint(col1 + 1, col2 - 1)]

    '''
    generation of 3 rectangle type feature
    '''
    def generate_3_rectangle (self):
        pass

    '''
    generation of 4 checkerboard type feature
    '''
    def generate_4_checkerboard (self):
        pass

    '''
    im -> Image to apply the feature on
    '''
    def apply_on_image (self, im):
        if self.type == 0:
            return self.apply_2_rectangle (im)
        # elif self.type == 1:
        #     self.apply_2_rectangle ()
        # elif self.type == 2:
        #     self.apply_2_rectangle ()

    '''
    im -> Image to apply feature on
    '''
    def apply_2_rectangle (self, im):
        topLeft = self.corners[0]
        bottomRight = self.corners[1]

        topMiddle = [topLeft[0], self.borders[0]]
        bottomMiddle = [bottomRight[0], self.borders[0]]

        leftRect = im.get_rect (topLeft, bottomMiddle)
        rightRect = im.get_rect (topMiddle, bottomRight)

        return leftRect - rightRect
    
    def package (self):
        return {
            "im_size" : self.im_size,
            "type" : self.type,
            "corners" : self.corners,
            "borders" : self.borders
        }

        


