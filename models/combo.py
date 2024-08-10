
import xgboost as xgb

def generate_combo_prediction(clf, preds):
    x_test = preds.reshape(-1, 1)
    # Generate prediction
    d_test = xgb.DMatrix(x_test)
    p_test = clf.predict(d_test)

    return p_test
