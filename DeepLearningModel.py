import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score \
    , fbeta_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score \
    , roc_curve

# Import the dataset
df_full = pd.read_csv('C://Users//Marcell//Downloads//creditcardfraud//creditcard.csv')

# Print out the first 5 row of the data set
print(df_full.head(5))

# Count the number of samples for each class (In this case we have 2 classes)
print(df_full.Class.value_counts())

# It is obvious that this data set is highly unbalance
# It Easier to sort the dataset by "class" for stratisfied sampling
df_full.sort_values(by='Class', ascending=False, inplace=True)

# Drop out the entire "Time" column
df_full.drop('Time', axis=1, inplace=True)

# Assign the first "3000" samples to new dataframe(data range)
df_sample = df_full.iloc[:3000, :]

# Count the number of samples for each class again(In this case we have two classes)
print(df_sample.Class.value_counts())

# Radomizes data to remove biasness
from sklearn.utils import shuffle

shuffle_df = shuffle(df_sample, random_state=42)

# Split the data set into two dataframe "train" & "test"
df_train = shuffle_df[0:2400]
df_test = shuffle_df[2400:]
# Input and output for data for test and train data
train_feature = np.array(df_train.values[:, 0:29])
train_label = np.array(df_train.values[:, -1])
test_feature = np.array(df_test.values[:, 0:29])
test_label = np.array(df_test.values[:, -1])

# Print out the size of train dataframe "should be of size 2400x29"
print(train_feature.shape)

# Print out the size of test dataframe "should be of size 2400x1"
print(train_label.shape)

# Standardize/Normalization the features columns to increase the training speed

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train_feature)
train_feature_trans = scaler.transform(train_feature)
test_feature_trans = scaler.transform(test_feature)


# A function to plot the learning curve
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


# Constructing the CNN & training phase
# Select the type of the model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt
model = Sequential()

# Add the first Dense layer with 200 neuron units and ReLu activation function
model.add(Dense(units=200,
                input_dim=29,
                kernel_initializer='uniform',
                activation='relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))

# Add the second Dense layer with 200 neuron units and ReLu activation function
model.add(Dense(units=200,
                kernel_initializer='uniform',
                activation='relu'))

# Add Dropout to prevent overfitting
model.add(Dropout(0, 5))

# Add the output layer with 1 neuron units and Sigmoid activation function
model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

# Print out the model summary
print(model.summary())

# Adam as a optimization function, and to optimize the Accuracy matrix
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Fit the model by pass 'train_feature_trans' as input X, 'train_lable'
# number of epochs = 200 and batch size = 500
train_history = model.fit(x=train_feature_trans, y=train_label
                          , validation_split=0.8, epochs=200
                          , batch_size=500, verbose=2)
# Print out the accuracy curves for training and validation sets
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# Print out the loss curves for training and validation sets

# Evaluation phase

# Use the testing set to evaluate the model
scores = model.evaluate(test_feature_trans, test_label)

# Print out the accuracy
print('\n')
print('Accuracy=', scores[1])

prediction = model.predict_classes(test_feature_trans)

df_ans = pd.DataFrame({'Real Class' :test_label})
df_ans['Prediction'] = prediction

df_ans['Prediction'].value_counts()
df_ans['Real Class'].value_counts()

cols = ['Real_Class_1', 'Real_Class_0']  # Gold standard
rows = ['Prediction_1', 'Prediction_0']  # Diagnostic tool (our prediction)

B1P1 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class'] == 1)])
B1P0 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class']) == 1])
B0P1 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class']) == 0])
B0P0 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class']) == 0])

conf = np.array([[B1P1, B0P1], [B1P0, B0P0]])
df_cm = pd.DataFrame(conf, columns=[i for i in cols], index=[i for i in rows])
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df_cm, annot=True, ax=ax, fmt='d')
plt.show()

# Making x label be on top is common in textbooks
ax.xaxis.set_ticks_position('top')

print('total test case number:' , np.sum(conf))


def model_efficacy(conf):
    total_num = np.sum(conf)
    sen = conf[0][0] / (conf[0][0] + conf[1][0])
    spe = conf[1][1] / (conf[1][0] + conf[1][1])
    false_positive_rate = conf[0][1] / (conf[0][1] + conf[1][1])
    false_negative_rate = conf[1][0] / (conf[0][0] + conf[1][0])

    print('total_num: ', total_num)
    print('G1P1: ', conf[0][0])  # G = gold standard; P = prediction
    print('G0P1: ', conf[0][1])
    print('G1P0: ', conf[1][0])
    print('G0P0: ', conf[1][1])
    print('##########################')
    print('sensitivity: ', sen)
    print('specificity: ', spe)
    print('false_positive_rate: ', false_positive_rate)
    print('false_negative_rate: ', false_negative_rate)

    return total_num, sen, spe, false_positive_rate, false_negative_rate


model_efficacy(conf)