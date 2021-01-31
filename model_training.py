from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

def logisticRegressionPrediction(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(solver='liblinear',
                                penalty='l1',
                                random_state=42)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return (y_pred, accuracy)
