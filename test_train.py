import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline # Note: Using Pipeline, not make_pipeline for grid search clarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# 1. Load Data
df = pd.read_csv("aita_cleaned.csv")
df['text'] = df['text'].fillna('').astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['is_asshole'], test_size=0.2, random_state=42, stratify=df['is_asshole']
)

# 2. Setup the "Smart" Stop Words again
my_stop_words = list(set(ENGLISH_STOP_WORDS) - {'not', 'no', 'nor', 'neither', 'never', 'none'})
my_stop_words.extend(['edit', 'update', 'formatting', 'mobile', 'found', 'title'])

# 3. Define the Pipeline (We give names 'tfidf' and 'svc' so we can talk to them later)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=my_stop_words)),
    ('svc', LinearSVC(class_weight='balanced', dual='auto', random_state=42))
])

# 4. Define the Grid (The options we want the computer to test)
param_grid = {
    # Try different vocabulary sizes
    'tfidf__max_features': [5000, 10000, 20000],
    
    # Try 1-word vs 2-word phrases vs 3-word phrases
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    
    # Try stricter vs looser regularlization (The "C" value)
    # C=0.1 (Simple model), C=1.0 (Default), C=10 (Complex model)
    'svc__C': [0.1, 0.5, 1.0, 10]
}

# 5. Run the Search
print("Running Grid Search... (This will take a few minutes)")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 6. The Results
print(f"\nBest Parameters Found: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

print("\n--- Final Evaluation on Test Set ---")
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
print(classification_report(y_test, predictions, target_names=['NTA', 'YTA']))
