"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        topList = Counter(y).most_common(2)
        length = topList[0][1]+topList[1][1]
        self.probabilities_ = (topList[1][1] if topList[0][0]==0 else topList[0][1])/length

        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)

        y = np.random.choice([0,1],X.shape[0],p=[1-self.probabilities_, self.probabilities_])

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)

    train_error = 0
    test_error = 0

    train_error_sum = 0
    test_error_sum = 0

    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=i)
        clf.fit(X_train,y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_error_temp = 1- metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        test_error_temp = 1- metrics.accuracy_score(y_test, y_pred_test, normalize=True)
        train_error_sum += train_error_temp
        test_error_sum += test_error_temp

    train_error = train_error_sum/ntrials
    test_error = test_error_sum/ntrials


    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    #for i in range(d) :
        #plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf_r = RandomClassifier()
    clf_r.fit(X, y)
    y_pred_r = clf_r.predict(X)
    train_rand_error = 1- metrics.accuracy_score(y, y_pred_r, normalize=True)
    print('\t-- training error: %.3f' % train_rand_error)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    decision_tree_clf = DecisionTreeClassifier(criterion="entropy")
    decision_tree_clf.fit(X, y)
    decision_tree_pred = decision_tree_clf.predict(X)
    decision_tree_train_error = 1- metrics.accuracy_score(y, decision_tree_pred, normalize=True)
    print('\t-- training error: %.3f' % decision_tree_train_error)
    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    kneighbor_clf_5 = KNeighborsClassifier()
    kneighbor_clf_3 = KNeighborsClassifier(n_neighbors=3)
    kneighbor_clf_7 = KNeighborsClassifier(n_neighbors=7)
    kneighbor_clf_5.fit(X, y)
    kneighbor_clf_3.fit(X, y)
    kneighbor_clf_7.fit(X, y)
    kneighbor_pred_5 = kneighbor_clf_5.predict(X)
    kneighbor_pred_3 = kneighbor_clf_3.predict(X)
    kneighbor_pred_7 = kneighbor_clf_7.predict(X)
    kneighbor_train_error_5 = 1- metrics.accuracy_score(y, kneighbor_pred_5, normalize=True)
    kneighbor_train_error_3 = 1- metrics.accuracy_score(y, kneighbor_pred_3, normalize=True)
    kneighbor_train_error_7 = 1- metrics.accuracy_score(y, kneighbor_pred_7, normalize=True)
    print('\t-- training error (k=5): %.3f' % kneighbor_train_error_5)
    print('\t-- training error (k=3): %.3f' % kneighbor_train_error_3)
    print('\t-- training error (k=7): %.3f' % kneighbor_train_error_7)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    majority_error = error(MajorityVoteClassifier(), X, y)
    random_error = error(RandomClassifier(), X, y)
    decision_tree_error = error(DecisionTreeClassifier(criterion="entropy"), X, y)
    kneighbor_error = error(KNeighborsClassifier(), X, y)
    print('\t-- error (majority training): %.3f' % majority_error[0])
    print('\t-- error (majority testing): %.3f' % majority_error[1])
    print('\t-- error (random training): %.3f' % random_error[0])
    print('\t-- error (random testing): %.3f' % random_error[1])
    print('\t-- error (decisiontree training): %.3f' % decision_tree_error[0])
    print('\t-- error (decisiontree testing): %.3f' % decision_tree_error[1])
    print('\t-- error (kneighbor training): %.3f' % kneighbor_error[0])
    print('\t-- error (kneighbor testing): %.3f' % kneighbor_error[1])
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    score = []
    list1 = []
    list2 = []
    for i in range(1,50,2):
        knn = KNeighborsClassifier(n_neighbors = i)
        cv_score = cross_val_score(knn, X, y, cv=10)
        avg = sum(cv_score)/len(cv_score)
        score.append([i,avg])
        list1.append(i)
        list2.append(1-avg)

    best_score = score[0]
    for k in range(len(score)):
        if score[k][1]>best_score[1]:
            best_score=score[k]
    print('\t-- best value of k is: %d' % best_score[0])

    """
    plt.figure()
    plt.plot(list1,list2)
    plt.xlabel('k')
    plt.ylabel('validation error')
    plt.show()
    """
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    list3 = []
    list4 = []
    list5 = []
    for i in range(1,21):
        clf_d = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        error_d = error(clf_d, X, y)
        list3.append(i)
        list4.append(error_d[0])
        list5.append(error_d[1])
    """
    plt.figure()
    plt.plot(list3,list4,'-b.', label='Training Error')
    plt.plot(list3,list5,'-r^', label='Test Error')
    plt.legend();
    plt.xlabel('depth')
    plt.ylabel('error')
    plt.show()
    """
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    trials_no = 100
    l1 = []
    error_train_d = []
    error_train_k = []
    error_test_d = []
    error_test_k = []
    for j in range(1, 11):
        error_train_d_total = 0
        error_train_k_total = 0
        error_test_d_total = 0
        error_test_k_total = 0
        for i in range(trials_no):
            X_train_large, X_test, y_train_large, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
            if j==10:
                X_train = X_train_large
                y_train = y_train_large
            else:
                X_train, X_dump, y_train, y_dump = train_test_split(X_train_large, y_train_large, test_size=1-0.1*j, random_state=i)
            clf_d = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            clf_d.fit(X_train, y_train)
            clf_k = KNeighborsClassifier(n_neighbors = 7)
            clf_k.fit(X_train, y_train)
            train_d = clf_d.predict(X_train)
            train_k = clf_k.predict(X_train)
            test_d = clf_d.predict(X_test)
            test_k = clf_k.predict(X_test)
            error_train_d_temp = 1- metrics.accuracy_score(y_train, train_d, normalize=True)
            error_train_k_temp = 1- metrics.accuracy_score(y_train, train_k, normalize=True)
            error_test_d_temp = 1- metrics.accuracy_score(y_test, test_d, normalize=True)
            error_test_k_temp = 1- metrics.accuracy_score(y_test, test_k, normalize=True)
            error_train_d_total += error_train_d_temp
            error_train_k_total += error_train_k_temp
            error_test_d_total += error_test_d_temp
            error_test_k_total += error_test_k_temp
        error_train_d.append(error_train_d_total/trials_no)
        error_train_k.append(error_train_k_total/trials_no)
        error_test_d.append(error_test_d_total/trials_no)
        error_test_k.append(error_test_k_total/trials_no)
        l1.append(j)

    plt.figure()
    plt.plot(l1,error_train_d,'-b.', label='tree - Training Error')
    plt.plot(l1,error_test_d,'-b^', label='tree - Test Error')
    plt.plot(l1,error_train_k,'-r.', label='knn  - Training Error')
    plt.plot(l1,error_test_k,'-r^', label='knn  - Test Error')
    plt.legend();
    plt.xlabel('Training Set Proportion')
    plt.ylabel('error')
    plt.show()

    best_score = []
    best_p = [1,1,1,1];
    best_score.append(error_train_d[0])
    best_score.append(error_test_d[0])
    best_score.append(error_train_k[0])
    best_score.append(error_test_k[0])
    for k in range(10):
        if error_train_d[k]<best_score[0]:
            best_p[0]=l1[k]
        if error_test_d[k]<best_score[1]:
            best_p[1]=l1[k]
        if error_train_k[k]<best_score[2]:
            best_p[2]=l1[k]
        if error_test_k[k]<best_score[3]:
            best_p[3]=l1[k]
    print('\t-- best proportion of training d is: %d' % best_p[0])
    print('\t-- best proportion of testing d is: %d' % best_p[1])
    print('\t-- best proportion of training k is: %d' % best_p[2])
    print('\t-- best proportion of testing k is: %d' % best_p[3])

    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
