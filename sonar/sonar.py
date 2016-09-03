import numpy
import pandas
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

start_time = datetime.now()
print(str(datetime.now()))
seed = 7
numpy.random.seed(seed)
df = pandas.read_csv("sonar.all-data.txt", header=None)
#df.describe()
ds = df.values
X = ds[:,0:60].astype(float)
Y = ds[:,60]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#df

def create_baseline():
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal',activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0 )
kfold = StratifiedKFold(y=encoded_Y, n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

print(str(datetime.now()))
print(str(datetime.now() - start_time))