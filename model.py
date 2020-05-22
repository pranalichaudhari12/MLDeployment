import pandas as pd
import pickle

df = pd.read_csv("iris.csv")
from sklearn.linear_model import LogisticRegression as LR
model = LR()
X = df.drop("species",1)
y = df["species"]
log = LR()
log.fit(X,y)

pickle.dump(log, open('Log.pkl', 'wb'))
model = pickle.load(open('Log.pkl','rb'))