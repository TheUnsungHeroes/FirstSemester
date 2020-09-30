import sys
import getopt
import os
import math
import operator
import collections



def update_count(counter1, counter2):
  for i in counter2:
    if i in counter1.keys():
      counter1[i] += counter2[i]
    else:
      counter1[i] = counter2.get(i)

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.BEST_MODEL = False
    #self.stopList = set(self.readFile('data/english.stop'))
    self.numFolds = 10
    self.posCount = 0
    self.negCount = 0
    self.posCountWord = 0
    self.negCountWord = 0
    self.posPrior = 0
    self.negPrior = 0
    self.posDict = {}
    self.negDict = {}
    self.totalDict = {}
    self.cardinal = 0
    self.posLikelihood = {}
    self.negLikelihood = {}

    # TODO: Add any data structure initialization code here


  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  # If the BEST_MODEL flag is true, include your new features and/or heuristics that
  # you believe would be best performing on train and test sets. 
  #
  # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the 
  # other two are meant to be off. That said, if you want to include stopword removal
  # or binarization in your best model, write the code accordingl
  # 
  # Hint: Use filterStopWords(words) defined below
  
  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    # Write code here:

    total_count = self.posCount + self.negCount
    self.posPrior = math.log(self.posCount / total_count)
    self.negPrior = math.log(self.negCount / total_count)

    ## CHECK FOR OVERLAPS:

    posDict_copy = {x:0 for x in self.posDict}
    negDict_copy = {x:0 for x in self.negDict}
    
    ## FIND NON-OVERLAPS
    self.posLikelihood = self.posDict.copy()
    update_count(self.posLikelihood, negDict_copy)

    self.negLikelihood = self.negDict.copy()
    update_count(self.negLikelihood, posDict_copy)

    sum_pos = sum(self.posLikelihood.values())
    sum_neg = sum(self.negLikelihood.values())
    cardinal = len(self.totalDict)

    ## LAPLACE SMOOTHING:
    self.posLikelihood = {x:math.log((self.posLikelihood.get(x)+1)/(sum_pos + cardinal)) for x in self.posLikelihood}
    self.negLikelihood = {x:math.log((self.negLikelihood.get(x)+1)/(sum_neg + cardinal)) for x in self.negLikelihood}

    all_words = self.totalDict.keys()
    pos_value = self.posPrior
    neg_value = self.negPrior

    for category in ("pos", "neg"):
        for word in words:
            if category == "pos":
                if word in all_words:
                    pos_value += self.posLikelihood.get(word)
            else:
                if word in all_words:
                    neg_value += self.negLikelihood.get(word)

    if pos_value > neg_value: 
      return 'pos' 
    else:
      return 'neg'
  
  
  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the NaiveBayes class.
     * Returns nothing
    """
    ## REMOVE PUNCTUATION:
    words = [word for word in words if word not in ["!", ",", ".", "(", ")", "?", "-", "'", '"', ":", ";"]]
    
    if self.BOOLEAN_NB == True:
      words = [word for word in set(words)]
    
    #word_count = collections.Counter(words)
    

    if klass == "pos":
      self.posCount += 1
      for word in words:
        self.posDict[word] = self.posDict.get(word, 0) + 1
        self.posCountWord += 1
        self.totalDict[word] = self.totalDict.get(word, 0) + 1
      #self.posPrior = math.log(self.posCount / (self.posCount + self.negCount))
      #update_count(self.posDict, word_count) #update_count(self.posDict, word_count)
    else:
      self.negCount += 1
      for word in words:
        self.negDict[word] = self.negDict.get(word, 0) + 1
        self.negCountWord += 1
        self.totalDict[word] = self.totalDict.get(word, 0) + 1
      #self.negPrior = math.log(self.negCount / (self.posCount + self.negCount))
    #   update_count(self.negDict, word_count) #update_count(self.negDict, word_count)
    
    # update_count(self.totalDict, word_count)
    # #self.cardinal = len(self.totalDict)

    # ## REVERT BACK TO DICTS:
    # posDict_copy = collections.Counter({x:0 for x in dict(self.posDict)})
    # negDict_copy = collections.Counter({x:0 for x in dict(self.negDict)})
    
    # ## FIND NON-OVERLAPS
    # self.posLikelihood = self.posDict.copy()
    # update_count(self.posLikelihood, negDict_copy)

    # self.negLikelihood = self.negDict.copy()
    # update_count(self.negLikelihood, posDict_copy)

    # new_pos = dict(self.posLikelihood)
    # new_neg = dict(self.negLikelihood)

    # sum_pos = sum(new_pos.values())
    # sum_neg = sum(new_neg.values())
    # cardinal = len(self.totalDict)

    # ## LAPLACE SMOOTHING:
    # self.posLikelihood = {x:math.log((new_pos.get(x)+1)/(sum_pos + cardinal)) for x in new_pos}
    # self.negLikelihood = {x:math.log((new_neg.get(x)+1)/(sum_neg + cardinal)) for x in new_neg}



      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      yield split

  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      guess = self.classify(words)
      labels.append(guess)
    return labels
  
  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = [] 
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1: 
      print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName)) 
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName)) 
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) )
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  classifier.BEST_MODEL = BEST_MODEL
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print(classifier.classify(testFile))
    
def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  BEST_MODEL = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  elif ('-m','') in options:
    BEST_MODEL = True
  
  if len(args) == 2 and os.path.isfile(args[1]):
    classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
  else:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)

if __name__ == "__main__":
    main()

# nb = NaiveBayes()
# text_test = nb.readFile(r"C:\Users\emilr\OneDrive - Aarhus universitet\Uni\CogSci - Master's-DESKTOP-TNA0AED\Study Group Exercises\FirstSemester\NLP\IMDB_reviews-master\arch_files\data\imdb1\pos\cv000_29590.txt")
# nb.addExample("pos", text_test)

# test_1 = collections.Counter({"a":1, "b":1})
# test_2 = collections.Counter({"a":1, "d":0})

# test_1.update(test_2)

# test_1

testy = collections.Counter({})
testy.update(collections.Counter({"a":2, "b":2}))

update_count(testy, collections.Counter({"a":3, "b":4}))

testy