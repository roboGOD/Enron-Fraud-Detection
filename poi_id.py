#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from poi_email_preprocess import dump_email_data, email_preprocessor
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### You will need to use more features

#features_list = ['poi', 'salary', 'bonus', 'deferral_payments','total_payments','exercised_stock_options',
#'restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances',
#'from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person','other']

### Best Features for Balanced Precision and Recall 
features_list = ['poi', 'salary', 'bonus', 'total_payments', 'exercised_stock_options',
'expenses','deferred_income','long_term_incentive', 'poi_interactions', 'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Fetch and Preprocess Email Data

#dump_email_data(data_dict)
#email_preprocessor(data_dict=data_dict)

with open("emails_preprocessed.pkl", "rb") as fOpen:
    emails_preprocessed = pickle.load(fOpen)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

### Task 3: Create new feature(s)
#import numpy as np 
to_pois = []
from_pois = []
words_from_emails = []
for key in sorted(data_dict.keys()):
    #print "\n",key, "\n"
    prev = emails_preprocessed['DERRICK JR. JAMES V']
    if key in emails_preprocessed:
        data_dict[key]['word_data'] = emails_preprocessed[key]
        words_from_emails.append(emails_preprocessed[key])
    else:
        data_dict[key]['word_data'] = prev
        words_from_emails.append(prev)
    for feature in data_dict[key]:
        #print "\t",feature,":",data_dict[key][feature]
        if feature != 'word_data':
            if data_dict[key][feature] < 0:
                data_dict[key][feature] = (-1)*data_dict[key][feature]


    from_poi = data_dict[key]['from_poi_to_this_person']
    to_poi = data_dict[key]['from_this_person_to_poi']
    total_to = data_dict[key]['from_messages']
    total_from = data_dict[key]['to_messages']


    if from_poi == "NaN":
        from_poi = 0
    if to_poi == "NaN":
        to_poi = 0
    if total_to == "NaN" or total_to == 0:
        total_to = 1
        to_poi = 0
    if total_from == "NaN" or total_from == 0:
        total_from = 1
        from_poi = 0

    to_ratio = float(to_poi)/ float(total_to)
    from_ratio = float(from_poi) / float(total_from)

    to_pois.append([to_ratio])
    from_pois.append([from_ratio])

    data_dict[key]['from_this_person_to_poi_ratio'] = to_ratio
    data_dict[key]['from_poi_to_this_person_ratio'] = from_ratio

    #print to_poi, ":", to_ratio
    #print from_poi, ":", from_ratio


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
transformed_to_pois = scaler.fit_transform(to_pois)
scaler = MinMaxScaler()
transformed_from_pois = scaler.fit_transform(from_pois)

i = 0
for key in data_dict:

    poi_interactions = transformed_from_pois[i][0] + transformed_to_pois[i][0]
    i += 1
    data_dict[key]['poi_interactions'] = poi_interactions



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features = np.concatenate((features, np.array(words_from_emails)), axis=1)
#print len(features[0])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
#from sklearn.neural_network import MLPClassifier




################################################################
### Trying Various Classifiers

#clf = DecisionTreeClassifier(min_samples_split = 2, max_depth = 11)



n_comp = 9000
classifier = GaussianNB()
classifier_name = 'gnb'
estimators = [('reduce_dim', PCA(n_components=n_comp)), (classifier_name, classifier)]
pipe = Pipeline(estimators)


### Parameters for MLPClassifier
#pipe.set_params(mlp__alpha=1e-5, mlp__hidden_layer_sizes=(5,2), mlp__max_iter=10000)

### Parameters for RandomForestClassifier
#pipe.set_params(rfc__min_samples_split=50, rfc__n_estimators=100)

### Parameters for DecisionTreeClassifier
#pipe.set_params(dtc__min_samples_split=2, dtc__max_depth = 5)

### Parameters for SVC kernel = 'rbf'
#param_grid = dict(svc__C=[1,10,100,1000,10000],svc__gamma=[0.0001,0.001,0.01,0.1,0.5,1], svc__kernel = ['poly','rbf'])
#clf = GridSearchCV(pipe, param_grid)

### Parameters for Linear SVC
#pipe.set_params(svc__kernel='linear')

### Parameters for AdaBoostClassifier
#pipe.set_params(adb__n_estimators=50)

#clf = GaussianNB()
clf = pipe



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

t0 = time()
clf.fit(features_train, labels_train)
print "Time Taken For Training:", round(time()-t0,3)
t0 = time()
predictions = clf.predict(features_test)
print "Time Taken For Predictions:", round(time()-t0,3),"\n"
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for prediction, truth in zip(predictions, labels_test):
    if prediction == 0 and truth == 0:
        true_negatives += 1
    elif prediction == 0 and truth == 1:
        false_negatives += 1
    elif prediction == 1 and truth == 0:
        false_positives += 1
    elif prediction == 1 and truth == 1:
        true_positives += 1
    else:
        print "Warning: Found a predicted label not == 0 or 1."
        print "All predictions should take value 0 or 1."
        print "Evaluating performance for processed predictions:"
        break

PERF_FORMAT_STRING = "\
\nAccuracy: {:>0.{display_precision}f}\nPrecision: {:>0.{display_precision}f}\n\
Recall: {:>0.{display_precision}f}\nF1: {:>0.{display_precision}f}\nF2: {:>0.{display_precision}f}"
#RESULTS_FORMAT_STRING = "\nTotal predictions: {:4d}\nTrue positives: {:4d}\nFalse positives: {:4d}\n\
#False negatives: {:4d}\nTrue negatives: {:4d}"
try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    #print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""
except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)