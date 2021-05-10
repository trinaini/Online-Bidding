# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Import libraries:
# %% [markdown]
# The dataset can be downloaded from (https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/data) or https://drive.google.com/drive/folders/1vXUVwal2sGqNn-uINu2D0lnor-Dljq-R?usp=sharing

# %%
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from matplotlib import pyplot as plt
# CHange root folder here as path to the data files
root_folder = '~/Desktop/facebook-recruiting-iv-human-or-bot/''

# %% [markdown]
# ### train_data & test_data exploratory analysis

# %%
# read train and test bidders data into a DataFrame
train = pd.read_csv(root_folder + 'train.csv')
test = pd.read_csv(root_folder + 'test.csv')


# %%
train.info()


# %%
# display the first 5 rows
print("The first 5 rows of Train Data is: \n")
train.head()

# %% [markdown]
# ### Descriptive statistics: Train Data

# %%
# Totals
total_bidders = len(train)
number_of_human_bidders = len(train[train['outcome'] == 0.0])
number_of_bot_bidders = total_bidders - number_of_human_bidders

# Proportions
human_bidders_proportion = number_of_human_bidders / total_bidders
bot_bidders_proportion = number_of_bot_bidders / total_bidders

# Statistics
print("Number of total bidders: {:,}".format(total_bidders))
print("Number of human bidders: {:,}".format(number_of_human_bidders))
print("Number of bot bidders: {:,}".format(number_of_bot_bidders))
print("Proportion of human bidders: {:.2%}".format(human_bidders_proportion))
print("Proportion of bot bidders: {:.2%}".format(bot_bidders_proportion))

#visualization 
_, ax = plt.subplots()

# Pie chart parameters
pie_data = [number_of_human_bidders, number_of_bot_bidders]
pie_labels = ('Humans', 'Bots')
pie_labels_explode_coefficients = (0, 0.175)

# Show the chart
ax.pie(pie_data, labels=pie_labels, autopct='%1.2f%%', shadow=False, explode=pie_labels_explode_coefficients)
plt.axis('equal')
plt.show()


# %%
# display the first 5 rows
print("The first 5 rows of Test Data is: \n")
test.head()

# %% [markdown]
# ### Bids exploratory analysis

# %%
# read the bids datasets 
bids = pd.read_csv(root_folder + 'bids.csv')
#bids.fillna('-', inplace=True) #replace NaN values with a dash as NaN are string categories we don't know
#bids = bids.sort_values(by=['auction', 'time']) #sort the bids of auction and time


# %%
bids.info()


# %%
# display the first 5 rows
print("The first 5 rows of bids Data is: \n")
bids.head()

# %% [markdown]
# ### Descriptive statistics: Bids Data

# %%
# Totals
total_bids = len(bids)
total_auctions = len(set(bids['auction']))
total_bidders_in_bids_dataframe = len(set(bids['bidder_id']))
total_devices = len(set(bids['device']))
total_countries = len(set(bids['country']))
total_ips = len(set(bids['ip']))
total_urls = len(set(bids['url']))
total_merchandise_categories = len(set(bids['merchandise']))

print("Number of bids: {:,}".format(total_bids))
print("Number of auctions: {:,}".format(total_auctions))
print("Number of total bidders in bids dataset: {:,}".format(total_bidders_in_bids_dataframe))
print("Number of devices: {:,}".format(total_devices))
print("Number of countries: {:,}".format(total_countries))
print("Number of IPs: {:,}".format(total_ips))
print("Number of URLs: {:,}".format(total_urls))
print("Number of merchandise categories: {:,}".format(total_merchandise_categories))


# %%
data_per_user = bids.groupby(['bidder_id'])

def get_user_statistics_per_feature(feature_column):
    return data_per_user[feature_column].nunique()

def print_user_statistics_per_feature(feature_name, feature_per_user):
    mean_feature_per_user = feature_per_user.mean()
    median_feature_per_user = feature_per_user.median()
    mode_feature_per_user = feature_per_user.mode()
    max_feature_per_user = feature_per_user.max()
    min_feature_per_user = feature_per_user.min()
    
    print("Average number of {} per user: {}".format(feature_name, mean_feature_per_user))
    print("Median of {} per user: {}".format(feature_name, median_feature_per_user))
    print("Mode of {} per user: {}".format(feature_name, mode_feature_per_user[0]))
    print("User with more {}: {}".format(feature_name, max_feature_per_user))
    print("User with less {}: {}".format(feature_name, min_feature_per_user))
    print("************************************************************")
    
    return feature_per_user

features_per_user = {}
for feat, column in [('auctions', 'auction'), ('bids', 'bid_id'), ('countries', 'country'), ('IPs', 'ip'), ('devices', 'device'), ('urls', 'url')]:
    features_per_user[column] = get_user_statistics_per_feature(column)
    print_user_statistics_per_feature(feat, features_per_user[column])
          
bids_per_user = features_per_user['bid_id'] / features_per_user['auction']
average_response_time_per_user = data_per_user['time'].apply(lambda x: x.diff().mean()).fillna(0)


# %%
# number of humans vs bots
y_pos = np.arange(len(pie_labels))
plt.bar(y_pos, pie_data, align='center', alpha=0.5, color=['green', 'red'])
plt.xticks(y_pos, pie_labels)
plt.ylabel('Count')
plt.title('Number of users per category')

plt.show()


# %%
# Distribution of auctions per user
plt.figure(figsize=(10,10))
plt.hist(features_per_user['auction'], bins='auto')
plt.yticks(range(0, 2800, 100))
plt.xticks(range(0, 1800, 100))
plt.title("Distribution of auctions per user")
plt.xlabel("Number of auctions")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# We can see that most of the users participate in less than 10 auctions as the distribution is skewed to the left. Also, it is visible that less than 50 users participate in 100 auctions or more.

# %%
# Distribution of bids per user
plt.plot(sorted(bids_per_user.values, reverse=True), 'r+--')
plt.title("Distribution of bids per user")
plt.xlabel("Number of bids")
plt.ylabel("Frequency")
plt.show()


# %%
# Distribution of countries per user
plt.figure(figsize=(10,10))
plt.hist(features_per_user['country'], bins='auto', color='red')
plt.title("Distribution of countries per user")
plt.xlabel("Number of countries")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# We can see that most of the data is located at the left side of the graph, which means most of the users bidded from less than two countries. 

# %%
# percentage breakdown of merchandise categories bid on, by bots
bids_1 = pd.merge(train, bids, on='bidder_id', how='left')
merch = bids_1[bids_1['outcome'] == 1].groupby('merchandise').size()
merch.sort_values()


# %%
_, ax = plt.subplots()
# Pie chart parameters
pie_data = merch
# Show the chart
ax.pie(pie_data, autopct='%1.2f%%', shadow=False)
plt.axis('equal')
plt.show()

# %% [markdown]
# Plot of percentage of merchandise categories bid by bots

# %%
# percentage breakdown of countries bots bid from
country = bids_1[bids_1['outcome'] == 1].groupby('country').size()
country.sort_values()


# %%
_, ax = plt.subplots()
# Pie chart parameters
pie_data = country
# Show the chart
ax.pie(pie_data, autopct='%1.2f%%', shadow=False)
plt.axis('equal')
plt.show()

# %% [markdown]
# Plot of percentage of countries bots bid from

# %%
# bids per auction per bidder id
bids_per_auction = bids_1.groupby(['auction', 'bidder_id']).size()
bids_per_auction = bids_per_auction.to_frame()
bids_per_auction.head()


# %%
# calculating time difference betwwen bid by a bidder
bids_1 = bids_1.sort_values(by=['time'])
bids_1['timediffs'] = bids_1.groupby('bidder_id')['time'].transform(pd.Series.diff)
bids_1.head()


# %%
# ip's to total bids ratio per bidder id 
ip_bids_ratio = bids_1.groupby('bidder_id')['ip'].nunique()/bids_1.groupby('bidder_id')['bid_id'].nunique()
ip_bids_ratio = ip_bids_ratio.to_frame()
ip_bids_ratio = ip_bids_ratio.reset_index()
ip_bids_ratio.head()


# %%
plt.scatter(ip_bids_ratio['bidder_id'],ip_bids_ratio[0], alpha=0.5)
plt.xticks([])
plt.show()

# %% [markdown]
# Scatter Plot showing number of ip to bids ratio
# %% [markdown]
# # Features creation

# %%
# Join 2 datasets together (-1 outcome meaning unknown i.e. test)
test['outcome'] = -1 # -1 makes test outcome as 'unknown'
bidders = pd.concat((train, test))

bidders.head()

# %% [markdown]
# .

# %%
#bidders['total_bids'] = 0
#for bidder in bidders['bidder_id']:
    #total_bids = bids[bids.bidder_id == bidder].count()[0]
    #bidders.loc[bidders[bidders.bidder_id == bidder].index, 'total_bids'] = total_bids


# %%
data_per_user = bids.groupby(['bidder_id'])

auctions_per_user = data_per_user['auction'].nunique().to_frame()
bids_per_user = data_per_user['bid_id'].count().to_frame()
countries_per_user = data_per_user['country'].nunique().to_frame()
ips_per_user = data_per_user['ip'].nunique().to_frame()
bids_per_auction_ratio_per_user = (bids_per_user['bid_id'] / auctions_per_user['auction']).to_frame()
average_response_time_per_user = data_per_user['time'].apply(lambda x: x.diff().mean()).fillna(0).to_frame()
# number of bids a user made per auction
bids_per_auction = bids.groupby(['auction', 'bidder_id']).size()
bids_per_auction = bids_per_auction.to_frame()

features_per_bidder = auctions_per_user.join(bids_per_user).join(countries_per_user).join(ips_per_user).join(
        bids_per_auction_ratio_per_user).join(average_response_time_per_user)
features_per_bidder.head()


# %%
def rename_features(df, name_mapping):
        df.rename(columns=name_mapping, inplace=True)
        return features_per_bidder


# %%
rename_features(features_per_bidder, {'auction': 'auctions_per_user',
                                       'bid_id': 'bids_per_user',
                                       'country': 'countries_per_user',
                                       'ip': 'ips_per_user',
                                       0: 'bids_auction_ratio',
                                       'time': 'average_response_time'})


# %%
newTrainData= train.join(features_per_bidder, how="left", on="bidder_id").fillna(0)
newTrainData.head()


# %%
newTestData= test.join(features_per_bidder, how="left", on="bidder_id").fillna(0)
newTestData.head()


# %%
del newTrainData['address']
del newTrainData['payment_account']

del newTestData['address']
del newTestData['payment_account']


# %%
newTrainData.head()


# %%
newTestData.head()

# %% [markdown]
# /////-----------------------NEW FEATURES----------------------------------------------------////////
# %% [markdown]
# ## Prepare & Split Data for Visualisation:

# %%
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# %%
# Prepare: TRAIN & TEST DATA
def prepare_train_and_test_data():
    train_data = newTrainData
    features = ['auctions_per_user', 'bids_per_user', 'countries_per_user', 'ips_per_user','bids_auction_ratio','average_response_time']
    labels = ['outcome']
    train_features = np.array(train_data[features])
    train_labels  = np.array(train_data[labels]).ravel()
    
    scaling features:
    train_features = preprocessing.MinMaxScaler().fit_transform(train_features)
    
    X_train, X_validation, y_train, y_validation = train_test_split(train_features, train_labels, test_size=0.33,random_state=42)
    
    data = {'X_train': X_train,
            'y_train': y_train,
            'X_validation': X_validation,
            'y_validation': y_validation}

    
    with open("train.p", "wb") as f:    #train.p will have prepared-train_data
        pickle.dump(data, f)
    
    test_data = newTestData
    test_features = np.array(test_data[features])
    test_features = preprocessing.MinMaxScaler().fit_transform(test_features)
    with open("test.p", "wb") as f:       #test.p will have prepared-test_data
        pickle.dump({'test_data': test,
                     'X_test': test_features}, f)


# %%
if __name__ == "__main__":
    prepare_train_and_test_data()

# %% [markdown]
# ## Training the classifiers with prepared TRAIN-DATA

# %%
from operator import itemgetter
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# %%
def train_model():
    model_classifiers = [{'name': 'Random Forest',
                              'classifier': RandomForestClassifier(),
                              'grid': {'n_estimators': (10, 20, 25, 50, 75, 100),
                                       'criterion': ('gini', 'entropy'),
                                       'min_samples_split': (2, 3, 5, 10),
                                       'min_samples_leaf': (1, 2, 5),
                                       'verbose': (1,),
                                       'n_jobs': (-1,)},
                              'score': 0.0},
                            
                         {'name': 'Gradient Boosting',
                              'classifier': GradientBoostingClassifier(),
                              'grid': {'loss': ('deviance', 'exponential'),
                                       'learning_rate': (0.1, 0.01, 0.005),
                                       'n_estimators': (10, 25, 50, 75, 100),
                                       'criterion': ('friedman_mse', 'mse', 'mae'),
                                       'min_samples_split': (2, 3, 5, 10),
                                       'min_samples_leaf': (1, 2, 5),
                                       'verbose': (1,)},
                              'score': 0.0},
                         
                         {'name': 'Decision Tree',
                              'classifier': DecisionTreeClassifier(),
                              'grid': {'criterion': ('gini', 'entropy'),
                                       'splitter': ('best', 'random'),
                                       'min_samples_split': (2, 3, 5, 10),
                                       'min_samples_leaf': (1, 2, 5)},
                              'score': 0.0},
                                 
                         {'name': 'SVC',
                              'classifier': SVC(),
                              'grid': {'kernel': ('rbf', 'poly', 'linear', 'sigmoid'),
                                       'degree': (3, 4, 5),
                                       'C': (1.0, 0.5, 0.001, 1.5)},
                              'score': 0.0},
                             
                         {'name': 'Logistic Regression',
                              'classifier': LogisticRegression(),
                              'grid': {'C': (1.0, 0.5, 0.001, 1.5)},
                              'score': 0.0},
                        
                         {'name': 'k-Nearest Neighbors',
                              'classifier': KNeighborsClassifier(),
                              'grid': {'n_neighbors': (2, 3, 5, 10, 50),
                                       'weights': ('uniform', 'distance'),
                                       'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                                       'p': (1, 2, 3)},
                              'score': 0.0}]
    
    with open("train.p", "rb") as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_validation = data['X_validation']
        y_validation = data['y_validation']
    
    for candidate in model_classifiers:
        print(candidate['name'])

        candidate['classifier'].fit(X_train, y_train)
        predictions = candidate['classifier'].predict(X_validation)
        candidate['score'] = roc_auc_score(y_validation, predictions)

        print("-----------------****************************-----------------------------")
        print("Score: ", candidate['score'])
        print("Accuracy: ", candidate['classifier'].score(X_validation, y_validation))
        print("-----------------****************************-----------------------------\n")

    top_classifier = sorted(model_classifiers, key=itemgetter('score'), reverse=True)[:3]
    pprint(top_classifier[0])
    #pprint(top_3_classifiers[1])
    #pprint(top_3_classifiers[2])
    
    with open("top_classifier.p", "wb") as f:   #top_3_classifiers.p = will have 3 best classifiers
        pickle.dump(top_classifier, f)


# %%
if __name__ == "__main__":
    train_model()

# %% [markdown]
# ## Submission: Using best model_classifiers on Test-features

# %%
from sklearn.metrics import accuracy_score

with open("train.p", "rb") as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_validation = data['X_validation']
        y_validation = data['y_validation']
        


gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, max_features='log2',
                                max_leaf_nodes=9)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_validation)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_pred, y_validation):.3f}")


# %%
bids = pd.read_csv(root_folder + 'bids.csv')
test = pd.read_csv(root_folder + 'test.csv')
test['outcome'] = -1
test = test.dropna()
bids = pd.merge(test, bids, on='bidder_id', how='left')

data_per_user = bids.groupby(['bidder_id'])

auctions_per_user = data_per_user['auction'].nunique().to_frame()
bids_per_user = data_per_user['bid_id'].count().to_frame()
countries_per_user = data_per_user['country'].nunique().to_frame()
ips_per_user = data_per_user['ip'].nunique().to_frame()
bids_per_auction_ratio_per_user = (bids_per_user['bid_id'] / auctions_per_user['auction']).to_frame()
average_response_time_per_user = data_per_user['time'].apply(lambda x: x.diff().mean()).fillna(0).to_frame()


features_per_bidder = auctions_per_user.join(bids_per_user).join(countries_per_user).join(ips_per_user).join(
        bids_per_auction_ratio_per_user).join(average_response_time_per_user)

features_per_bidder.head()

def rename_features(df, name_mapping):
        df.rename(columns=name_mapping, inplace=True)
        return features_per_bidder
    
rename_features(features_per_bidder, {'auction': 'auctions_per_user',
                                        'bid_id': 'bids_per_user',
                                        'country': 'countries_per_user',
                                        'ip': 'ips_per_user',
                                        0: 'bids_auction_ratio',
                                        'time': 'average_response_time'})


# %%
features_per_bidder = features_per_bidder.fillna(0)
features = ['auctions_per_user', 'bids_per_user', 'countries_per_user', 'ips_per_user','bids_auction_ratio','average_response_time']

X= features_per_bidder[features]

y_pred_test = gb.predict(X)


# %%
features_per_bidder['prediction'] = y_pred_test
a = features_per_bidder['prediction']
a.to_csv('submission.csv')


# %%
features_per_bidder['prediction'].head()


# %%



