from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))