from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("aita.csv", encoding='cp1252')

valid_labels = ['nta', 'yta']
df = df[df['verdict'].isin(valid_labels)]

label_map = {'nta': 0, 'yta': 1, 'nah': 2, 'esh': 3}
df['label'] = df['verdict'].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    df['title'],
    df['label'],
    test_size=0.2,
    random_state=42
)

model = make_pipeline(TfidfVectorizer(max_features=5000), LogisticRegression())

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(
    y_test,
    predictions,
    target_names=['nta', 'yta']))
