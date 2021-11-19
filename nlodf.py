import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pickle

pd.options.display.max_columns = None
pd.options.display.width = None

ufos = pd.read_csv('https://github.com/nzhdnva/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/data/ufos.csv?raw=true')
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'], 'Latitude': ufos['latitude'],
                     'Longitude': ufos['longitude']})

ufos.dropna(inplace=True)
# отбрасываем нулевые значения
ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
# оставляем наблюдения в промежутке минуты

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

"""
print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
"""

model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))