import os
import re
import logging
import math
from stop_words import get_stop_words


logging.basicConfig()
logger = logging.getLogger('Naive Bayes')
logger.setLevel(logging.INFO)

class MultinomialNB(object):
    def __init__(self, alpha = None):
        # laplace smoothing factor
        if alpha is None:
            self.alpha = 1
        else:
            self.alpha = alpha
        # attrs:{attrName: {class : counts}}
        self.attrs = {}
        # class: {class : counts}
        self.classes = {}
        # all possible words
        self.vocabulary = set()
        # accuracy
        self.accuracy = None
        logger.info('Naive bayes initiated')

    # Fit the training set in multinomial naive bayes
    def fit(self, train_path):
        logger.info('Fitting training set into the naive bayes model')
        self.classes = self.getClassesFreq(train_path)
        # extract word from each train file
        countsByClass = self.getWordCount(train_path, self.classes, self.attrs)
        self.laplaceSmoothing(countsByClass)
        return

    # Predict the class of test set using multinomial naive bayes
    def predict(self, test_path):
        logger.info('Labeling the test set')
        testClasses = self.getClass(test_path)
        expectlabels = {}
        for testClass in testClasses:
            for label in os.listdir(test_path + '/' + testClass):
                expectlabels[label] = testClass
        logger.info('Finished labeling of test set')

        labels = self.getExpectedClassForFolder(test_path)
        self.accuracy = self.getAccuracy(labels, expectlabels)
        return

    # Methods to get all labels under path to training set
    def getClass(self, train_path):
        logger.info('Getting class outputs under training path')
        classes = [line for line in os.listdir(train_path) if not line.startswith(".")]
        logger.info('Finished getting class outputs under training path: %d' % len(classes))
        return classes

    # Check whether a word is a valid word
    def isValid(self, word):
        hasNum = bool(re.search(r'\d', str(word)))
        hasDash = bool('_' in str(word) or '\'' in str(word) or '/' in str(word))
        stopwords = get_stop_words('english')
        return not hasNum and not hasDash and len(word) < 20 and len(word) > 2 and word not in stopwords

    # Calculate the prior prob of labels, P(Y)
    def getClassesFreq(self, path):
        logger.info('Extracting frequency of class outputs under %s' % path)
        classes = self.getClass(path)
        classFreq = {}
        total = 0;
        for classOutput in classes:
            classFreq[classOutput] = len(os.listdir(path + "/" + classOutput))
            total = total + classFreq[classOutput]

        for classOutput in classes:
            classFreq[classOutput] = float(classFreq[classOutput]) / float(total)

        logger.info('Finished extraction of frequency of class outputs from %s' % path)
        return classFreq

    # Count the occurrences of words in a single file
    def extractWordFromFile(self, pathToFile):
        logger.debug('Extracting words from file %s' % pathToFile)
        wordCounts = {}
        total = 0
        with open(pathToFile, 'rb') as file:
            for line in file:
                d = re.findall(r"[\w']+", str(line))
                for word in d:
                    if self.isValid(word):
                        total = total + 1
                        word = word.lower()
                        wordCounts[word] = wordCounts.get(word, 0) + 1
        logger.debug('Finished extracting words from file %s' % pathToFile)
        return total, wordCounts

    # Count the occurrence of words in one category or label
    def getWordCount(self, path, classes, attrs):
        logger.info('Calculating word counts from training set under %s' % path)
        countsByClass = {}
        for str in classes:
            pathToFolder = path + '/' + str
            total = 0
            for file in os.listdir(pathToFolder):
                pathToFile = pathToFolder + '/' + file
                count, wordCounts = self.extractWordFromFile(pathToFile)
                total = total + count
                for word in wordCounts:
                    self.vocabulary.add(word)
                    if word not in attrs:
                        attrs[word] = {}
                    if str not in attrs[word]:
                        attrs[word][str] = 0
                    attrs[word][str] += wordCounts[word]
            countsByClass[str] = total
        logger.info('Finished extraction of word counts from training set under %s' % path)
        return countsByClass

    # Laplace smoothing
    def laplaceSmoothing(self, countsByClass):
        # laplace smoothing for naive bayes
        logger.info('Calculating conditional probabilities of words using laplace smoothing')
        for str in countsByClass:
            counts = countsByClass[str]
            for word in self.attrs:
                if str in self.attrs[word]:
                    self.attrs[word][str] = \
                        float(self.attrs[word][str] + 1) / float(counts + self.alpha * len(self.vocabulary))
                else:
                    self.attrs[word][str] = 1 / float(counts + self.alpha * len(self.vocabulary))
        logger.info('Finished calculation of conditional probabilites of words')
        return

    # Calculate the expected label of a test file
    def getExpectedClassForFile(self, testFilePath):
        total, wordCounts = self.extractWordFromFile(testFilePath)
        maxProb = None
        expectLabel = ''
        for label in self.classes:
            prob = math.log(self.classes[label])
            for word in wordCounts:
                if(word in self.attrs):
                    prob += math.log(self.attrs[word][label])
            if maxProb is None or prob > maxProb:
                maxProb = prob
                expectLabel = label
        return expectLabel

    # Calcuate expected labels of all test files given path to all test files
    def getExpectedClassForFolder(self, testFolderPath):
        labels = {}
        for label in os.listdir(testFolderPath):
            if not label.startswith('.'):
                labelPath = testFolderPath + '/' + label
                for file in os.listdir(labelPath):
                    labels[file] = self.getExpectedClassForFile(labelPath + '/' + file)
        return labels;

    # Calculate the accuracy of the multinomial naive bayes
    def getAccuracy(self, labels, expectLabels):
        match = 0
        for x in labels:
            if labels[x] == expectLabels[x]:
                match = match + 1
        return float(match) / float(len(labels))

if __name__ == '__main__':
    train_path = input('Path to training set: ')
    test_path = input('Path to test set: ')
    clf = MultinomialNB()
    clf.fit(train_path)
    clf.predict(test_path)
    print('Accuracy of the MNB is {0:.02f}%'.format(clf.accuracy * 100))