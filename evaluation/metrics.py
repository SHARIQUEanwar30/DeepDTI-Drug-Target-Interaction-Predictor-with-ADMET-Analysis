from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(y_true, y_pred, y_prob):

    print("Accuracy:", accuracy_score(y_true,y_pred))
    print("F1:", f1_score(y_true,y_pred))
    print("ROC-AUC:", roc_auc_score(y_true,y_prob))
