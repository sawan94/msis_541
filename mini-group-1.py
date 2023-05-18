
import nltk
import random

from nltk.corpus import subjectivity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings



def warn(*args, **kwargs):
    pass
warnings.warn = warn

def tk(doc):
    return doc

def execute_models(**params):
    print(params)
    n_instances = 200
    subj_docs = [(sent, 'subj') for sent in
                 subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj') for sent in
                subjectivity.sents(categories='obj')[:n_instances]]

    train_subj_docs = subj_docs[:150]
    test_subj_docs = subj_docs[150:200]
    train_obj_docs = obj_docs[:150]
    test_obj_docs = obj_docs[150:200]
    # Pool
    training_docs = train_subj_docs + train_obj_docs
    testing_docs = test_subj_docs + test_obj_docs
    # Separate X and Label
    training_x = [i[0] for i in training_docs]
    testing_x = [i[0] for i in testing_docs]
    training_c = [i[1] for i in training_docs]
    testing_c = [i[1] for i in testing_docs]



    vec = TfidfVectorizer(analyzer='word', tokenizer=tk,
                          preprocessor=tk, token_pattern=None, stop_words='english', **params)
    vec.fit(training_x)
    training_x = vec.transform(training_x)
    testing_x = vec.transform(testing_x)

    # logit model
    Logitmodel = LogisticRegression()
    Logitmodel.fit(training_x, training_c)
    y_pred_logit = Logitmodel.predict(testing_x)
    acc_logit = accuracy_score(testing_c, y_pred_logit)
    print("Logit model Accuracy:: {:.2f}%".format(acc_logit * 100))

    # Naive Bayes
    NBmodel = MultinomialNB()
    # training
    NBmodel.fit(training_x, training_c)
    y_pred_NB = NBmodel.predict(testing_x)
    # evaluation
    acc_NB = accuracy_score(testing_c, y_pred_NB)
    print("Naive Bayes model Accuracy::{:.2f}%".format(acc_NB*100))


    #svm
    SVMmodel = LinearSVC()
    # training
    SVMmodel.fit(training_x, training_c)
    y_pred_SVM = SVMmodel.predict(testing_x)
    # evaluation
    acc_SVM = accuracy_score(testing_c, y_pred_SVM)
    print("SVM model Accuracy:{:.2f} %".format(acc_SVM*100))

    DTmodel = DecisionTreeClassifier()
    RFmodel = RandomForestClassifier(n_estimators=50, max_depth=3,
    bootstrap=True, random_state=0) ## number of trees and number of layers/depth
    # training
    DTmodel.fit(training_x, training_c)
    y_pred_DT = DTmodel.predict(testing_x)
    RFmodel.fit(training_x, training_c)
    y_pred_RF = RFmodel.predict(testing_x)
    # evaluation
    acc_DT = accuracy_score(testing_c, y_pred_DT)
    print("Decision Tree Model Accuracy: {:.2f}%".format(acc_DT*100))
    acc_RF = accuracy_score(testing_c, y_pred_RF)
    print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))

    print("******")
    print("")



if __name__ == '__main__':
    execute_models(min_df=3,ngram_range=(1, 1))
    execute_models(min_df=3,ngram_range=(1, 2))
    execute_models(min_df=2,ngram_range=(1, 2))
