import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv

############# Generate training data #############

print('Please Wait - Model Training')

data1_x = pd.read_csv('wili-2018/x_train.txt', sep='delimiter', header=None, dtype=str, encoding='utf8', engine='python')
data1_x = np.array(data1_x)
data1_y = np.reshape(np.loadtxt('wili-2018/y_train.txt',delimiter=',', dtype=str),(len(data1_x),1))

data2_x = pd.read_csv('wili-2018/x_test.txt', sep='delimiter', header=None, dtype=str, encoding='utf8', engine='python')
data2_x = np.array(data2_x)
data2_y = np.reshape(np.loadtxt('wili-2018/y_test.txt',delimiter=',' , dtype=str),(len(data2_x),1))

data_x = np.vstack((data1_x,data2_x))
data_y = np.vstack((data1_y,data2_y))

Persian = []
Arabic = []
German = []
Turkish = []
English = []
French = []

for i in range(len(data_x)):
    if data_y[i] == 'fas':
        Persian.append(str(data_x[i]))
        
    if data_y[i] == 'ara':
        Arabic.append(str(data_x[i]))
        
    if data_y[i] == 'deu':
        German.append(str(data_x[i]))
        
    if data_y[i] == 'eng':
        English.append(str(data_x[i]))
        
    if data_y[i] == 'tur':
        Turkish.append(str(data_x[i]))
        
    if data_y[i] == 'fra':
        French.append(str(data_x[i]))
        

dataset_x = np.hstack((English,Arabic,Persian,German,French,Turkish))
dataset_y = np.empty((len(dataset_x),1), np.dtype('U25'))
dataset_y[0:1000] = 'English'
dataset_y[1000:2000]= 'Arabic'
dataset_y[2000:3000]= 'Persian'
dataset_y[3000:4000]= 'German'
dataset_y[4000:5000]= 'French'
dataset_y[5000:6000]= 'Turkish'

###########################################

############# Preprocess data #############

data_processed = []
numbers = ['0','1','2','3','4','5','6','7','8','9']
punctuation = ['`','-','=',',','[',']',';','/','.',',','~','!','@','#','$','%','^','&','*',
               '(',')','_','+','|','{','}',':','"','?','<','>','}','’','“','”' , '\'']
for text in dataset_x:
    for character in numbers:
        text = text.replace(character, "")
    for character in punctuation:
        text = text.replace(character, "")   
    text = text.lower()
    text = text.strip()
        # appending to data_list
    data_processed.append(text)
        
cv = CountVectorizer(max_features=100)
new_data_x = cv.fit_transform(data_processed).toarray()

###########################################

############# Multinomial Naive Bayes implementation #############

class Multinomial_NB():
    
    def fit(self, train_x, train_y):
        ####### Training the Model by Calculating Prior and Likelihood #######
        m, n = train_x.shape
        self.classes = np.unique(train_y)
        num_classes = len(self.classes)
        self.priors = np.zeros(num_classes)
        self.likelihoods = np.zeros((num_classes, n))
        alpha = 1
        
        for i, j in enumerate(self.classes):
            class_index = train_x[np.where(j == train_y)[0], :]
            self.priors[i] = class_index.shape[0] / m 
            self.likelihoods[i, :] = ((class_index.sum(axis=0)) + alpha) / (np.sum(class_index.sum(axis=0) + alpha))
            
    def predict(self, test_x):
        
        predicted_y = []
        for i in test_x:
        ####### Predicting Category by Calculating Posteriors #######
            posteriors = []
            for j, k in enumerate(self.classes):
                class_prior = np.log(self.priors[j])
                class_likelihoods = np.log(self.likelihoods[j,:]) * i
                class_posteriors = np.sum(class_likelihoods) + class_prior
                posteriors.append(class_posteriors)
        
            predicted_y.append(self.classes[np.argmax(posteriors)])
        return predicted_y

###########################################

############# Training Model #############

model = Multinomial_NB()
model.fit(new_data_x , dataset_y)

###########################################

############# Testing Model #############

# test = pd.read_csv('test/task1 - task1.csv', dtype=str  , encoding='utf8')
# test = np.array(test)
# test_processed = cv.transform(list(np.reshape(test,(len(test),)))).toarray()

# predicted_y = model.predict(test_processed)

# true_label = np.array(['Arabic','German','English','English','English','Persian','Persian','German','English',
#               'Persian','German','German','German','Arabic','Arabic','Arabic','English','Persian',
#               'German','German','Persian','German','Persian','Persian','Arabic','Arabic','English',
#               'Arabic','Persian','German','German','Persian','German','Arabic','Arabic','English',
#               'Persian','Persian','English','Arabic','German','Persian','Persian','Persian','Arabic',
#               'German','Arabic','German','English','English','German','Arabic','Arabic','English',
#               'Arabic','English','Arabic','English','Persian','Arabic','English','English','Persian',
#               'German','English','English','Persian','Persian','English','Arabic','Arabic','English',
#               'English','Persian','English','Persian','German','Arabic','German','Persian','Persian',
#               'German','Arabic','Arabic','Persian','English','English','German','German','German',
#               'English','German','German','Arabic','Persian','Arabic','Arabic','English','Persian','German'])
# accuracy = np.sum(true_label == np.array(predicted_y))/len(true_label)
# print('Accuracy is: ' , accuracy)

test_path = input('Please enter path of test set(Example: tests/test1.csv): ')
test = pd.read_csv(test_path, dtype=str  , encoding='utf8')
test = np.array(test)
test_processed = cv.transform(list(np.reshape(test,(len(test),)))).toarray()
predicted_y = model.predict(test_processed)

###########################################

############# Creating SCV File #############

header = ['Id','Category']
test_data = np.hstack((test,np.reshape(np.array(predicted_y),(len(predicted_y),1))))
with open('test_data.csv', mode='w' , encoding='UTF8' , newline='') as csv_file:
    
    writer = csv.writer(csv_file)
    writer.writerow(header)

    writer.writerows(test_data)
print('File saved successfully')        
###########################################    