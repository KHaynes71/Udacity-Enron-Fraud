#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pickle
import matplotlib.pyplot as plt 
import pprint

sys.path.append ("C:/ud120-projects-master/tools")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import learning_curve, GridSearchCV
from numpy import mean
from sklearn.model_selection import train_test_split
import numpy as np



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 


financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']
### Load the dictionary containing the dataset
#with open("C:/ud120-projects-master/tools/final_project_dataset.pkl", "rb") as data_file:
    #data_dict = pickle.load(data_file)


#my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)



### Load the dictionary containing the dataset
with open("C:/ud120-projects-master/tools/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

all_features = features_list + email_features + financial_features 
all_features.remove('email_address') 










# In[2]:


#Figuring out how many names and POIs are in the data set to start with

print('Number of employees in the Enron data set: %d' % (len(data_dict)))

poi_ct=0
for key, val in data_dict.items():
    if (val['poi'] == 1.0):
        poi_ct = poi_ct + 1
        
print ('POIs identified in the data: %d' % poi_ct)


# In[3]:


#exploring available keys

pprint.pprint(list(data_dict.keys()))


# In[4]:


'''Removing outlying keys based on manual examination of the dataset above- including The Travel Agency... and 'Total'. 
Additionaly, removing email addresses as they will not add prediction value.'''
email_features.remove('email_address') 
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('TOTAL')


# In[5]:


#Checking my work:
pprint.pprint(list(data_dict.keys()))


# In[6]:


pprint.pprint(all_features)


# In[7]:


data = featureFormat(data_dict, all_features, sort_keys = False)
poi=data[:,0]
salary = data[:,6]
bonus = data[:,10]

plt.scatter(salary[poi==1],bonus[poi==1],c='orange',s=10,label='POI')
plt.scatter(salary[poi==0],bonus[poi==0],c='green',s=10,label='Non-POI')

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.legend(loc='lower right')
plt.title("Salary and Bonus of POIs")
plt.show()


# In[8]:


data2 = featureFormat(data_dict, all_features, sort_keys = True)
poi=data2[:,0]
from_this_person_to_poi = data2[:,4]
from_poi_to_this_person = data2[:,2]

plt.scatter(from_this_person_to_poi[poi==1],from_poi_to_this_person[poi==1],c='red',s=30,label='POI')
plt.scatter(from_this_person_to_poi[poi==0],from_poi_to_this_person[poi==0],c='blue',s=30,label='Non-POI')

plt.xlabel("from_this_person_to_poi")
plt.ylabel("from_poi_to_this_person")
plt.legend(loc='lower right')
plt.title("Emails sent/received with POIs")
plt.show()


# In[9]:


#Print minimums and maximums to look for outliers
for feature in all_features:
    print (feature)
    feature = [item[feature] for k, item in 
    data_dict.items() if not item[feature] == "NaN"]
    print ('min is: %d' % min(feature))
    print ('max is: %d' % max(feature))


# In[10]:


outliers =[]
for key in data_dict.keys():
    if (data_dict[key]['deferral_payments']!='NaN')& (data_dict[key]['deferred_income']!='NaN'):
        if (int(data_dict[key]['deferred_income'])< 0):
            outliers.append(key)
print ("Outlier Names:",outliers)


# In[11]:


#Examining one outlier
pprint.pprint(data_dict['BAXTER JOHN C'])


# In[12]:


### Task 3: Create new feature(s)
def calcluatePercent(messages, allMessages):
    percent = 0
    if (messages == 'NaN' or allMessages == 'NaN'):
        return percent
    percent = messages / float(allMessages)
    return percent


def createNewFeatures(data_dict):
    for poi_name in data_dict:
        new_dict = data_dict[poi_name]
        new_dict['from_poi_to_this_person_ratio'] = calcluatePercent(new_dict['from_poi_to_this_person'],
                                                                   new_dict['to_messages'])
        new_dict['from_this_person_to_poi_ratio'] = calcluatePercent(new_dict['from_this_person_to_poi'],
                                                                   new_dict['from_messages'])
    return new_dict, ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio']



for entry in data_dict:

    data_point = data_dict[entry]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    percent_from_poi = calcluatePercent(from_poi_to_this_person, to_messages )
    data_point["percent_from_poi"] = percent_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    percent_to_poi = calcluatePercent( from_this_person_to_poi, from_messages )
    data_point["percent_to_poi"] = percent_to_poi
features_list_n = all_features
features_list_n =  features_list_n + ['percent_from_poi', 'percent_to_poi']
pprint.pprint (features_list_n)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_n, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[13]:


#Accidentally pulled in Email Address again, removing:
#features_list_n.remove('email_address') 
pprint.pprint (features_list_n)


# In[14]:


#Now I want to plot and look at my new features
data = featureFormat(data_dict, features_list_n, sort_keys = False)
poi=data[:,0]
percent_from_poi = data[:,20]
percent_to_poi = data[:,21]

plt.scatter(percent_from_poi[poi==1],percent_to_poi[poi==1],c='red',s=30,label='POI')
plt.scatter(percent_from_poi[poi==0],percent_to_poi[poi==0],c='blue',s=30,label='Non-POI')

plt.xlabel("percent_from_poi")
plt.ylabel("percent_to_poi")
plt.legend(loc='lower right')
plt.title("Percent of all messages sent to and from POIs")
plt.show()


# In[15]:


import pandas as pd
pd.set_option('display.max_rows', 10)
pd.DataFrame.from_dict(data_dict, orient='index',
                       columns=['poi',
 'to_messages',
 'from_poi_to_this_person',
 'from_messages',
 'from_this_person_to_poi',
 'shared_receipt_with_poi',
 'salary',
 'deferral_payments',
 'total_payments',
 'loan_advances',
 'bonus',
 'restricted_stock_deferred',
 'deferred_income',
 'total_stock_value',
 'expenses',
 'exercised_stock_options',
 'other',
 'long_term_incentive',
 'restricted_stock',
 'director_fees',
 'percent_from_poi',
 'percent_to_poi'])


# In[16]:


for feature in features_list_n:
    print (feature)
    feature = [item[feature] for k, item in 
    data_dict.items() if not item[feature] == "NaN"]
    print ('min is: %d' % min(feature))
    print ('max is: %d' % max(feature))


# In[17]:


def skipOne(elem):
    return elem[1]
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 5)
selector.fit(features, labels)
scores = zip(features_list_n[1:], selector.scores_)
sorted_scores = sorted(scores, key = skipOne, reverse = True)
pprint.pprint('SelectKBest scores: ')
pprint.pprint( sorted_scores)
all_features =  features_list + [(i[0]) for i in sorted_scores[0:20]]
pprint.pprint( all_features)
kBest_features = features_list + [(i[0]) for i in sorted_scores[0:10]]
pprint.pprint( 'KBest')
pprint.pprint( kBest_features)


# In[18]:


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[35]:


#from sklearn.model_selection import train_test_split
#features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
from time import time

def naive_bayes_clf(features_train, features_test, labels_train, labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print ("\ntraining time:", round(time()-t0, 3), "s")

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print ("predicting time:", round(time()-t0, 3), "s")
    accuracy = accuracy_score(pred, labels_test)
    print ('\naccuracy = {0}'.format(accuracy))

    return clf


def svm_clf(features_train, features_test, labels_train, labels_test):
    from sklearn.svm import SVC
    clf = SVC(kernel="linear", C=1000)
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print ("\ntraining time:", round(time()-t0, 3), "s")

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print ("predicting time:", round(time()-t0, 3), "s")
    accuracy = accuracy_score(pred, labels_test)
    print ('\naccuracy = {0}'.format(accuracy))

    return clf


def decision_tree_clf(features_train, features_test, labels_train, labels_test):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print ("\ntraining time:", round(time()-t0, 3), "s")

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print ("predicting time:", round(time()-t0, 3), "s")
    accuracy = accuracy_score(pred, labels_test)
    print ('\naccuracy = {0}'.format(accuracy))

    return clf

def adaboost_clf(features_train, features_test, labels_train, labels_test):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(learning_rate=1, algorithm='SAMME', n_estimators=23)
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print ("\ntraining time:", round(time()-t0, 3), "s")

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print ("predicting time:", round(time()-t0, 3), "s")
    accuracy = accuracy_score(pred, labels_test)
    print ('\naccuracy = {0}'.format(accuracy))

    return clf


# In[36]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#!/usr/bin/pickle

#clf = naive_bayes_clf(features_train, features_test, labels_train, labels_test)
#clf = svm_clf(features_train, features_test, labels_train, labels_test)
clf = decision_tree_clf(features_train, features_test, labels_train, labels_test)
#clf = adaboost_clf(features_train, features_test, labels_train, labels_test)







