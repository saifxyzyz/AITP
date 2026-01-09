from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import joblib

df = pd.read_csv("aita_cleaned.csv.gz")
df['text'] = df['text'].fillna('').astype(str)


X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['is_asshole'],
    test_size=0.5,
    random_state=42,
    stratify=df['is_asshole']
)

my_stop_words = list(set(ENGLISH_STOP_WORDS) - {'not', 'no', 'nor', 'neither',
                                                'never', 'none'})
my_stop_words.extend(['aita', 'yta', 'nta', 'esh', 'nah', 'wibta', 'iwbta',
                      'awta', 'update', 'iata', 'edit', 'title', 'mobile', 
                      'formatting', 'tl', 'dr'])

log_reg = LogisticRegression(class_weight='balanced', solver='sag', C=1.0, max_iter=1000, random_state=42)
linear_svm = LinearSVC(class_weight='balanced', dual='auto', C=0.1, random_state=42)
calibrated_svm = CalibratedClassifierCV(linear_svm, method='sigmoid', cv=3)
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=30000,
                              ngram_range=(1, 3),
                              stop_words=my_stop_words,
                              min_df=50,
                              max_df=0.7,
                              sublinear_tf=True
                              )),
    ('ensemble', VotingClassifier(
        estimators=[
            ('lr', log_reg),
            ('svm', calibrated_svm)
        ],
        voting='soft'
    ))
])


model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(
    y_test,
    predictions,
    target_names=['nta', 'yta']))
print("Saving model to 'aita_model.pkl'")
joblib.dump(model, 'aita_model.pkl')
