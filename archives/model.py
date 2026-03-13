import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold, cross_val_score
import ee, geemap
from geemap import ml
ee.Initialize()
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import shap
import numpy as np

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

data_preprocessing = "smote"
send_model = True
start_year = 2010
end_year = 2015
if __name__ == '__main__':
    data = pd.read_csv("./data/training_data_" + str(start_year) +  "_" + str(end_year) + "_plus.csv")

    X = data[["0_Rainfall_norm", "1_precipitation", "2_NDVI", "3_EVI", "4_ground_temp", "5_edge_distance", "6_elevation", "7_slope", "8_deforestation_density"]]
    y = data["9_deforestation"]
    print(y.unique())
    x_under, y_under = RandomUnderSampler(random_state = 42).fit_resample(X, y)
    x_over, y_over = RandomOverSampler(random_state = 42).fit_resample(X, y)
    x_smote, y_smote = SMOTE(random_state = 42).fit_resample(X, y)

    if data_preprocessing == "regular":
        """
        Regular Modeling
        """
        print("------------------REGULAR------------------")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=20, max_depth=5)
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        pred = model.predict(X_test)
        pred_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, pred_prob[:,1])
        accuracy = accuracy_score(y_test, pred)
        print(classification_report(y_test, pred))
        print(f'ROC-AUC Score: {roc_auc}')
        print(f'Accuracy: {accuracy}')
        print(f"Scores for each fold: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        print(f"Standard deviation of accuracy: {scores.std():.4f}")
        X_shap = X_train
    
    elif data_preprocessing == "undersampling":
        """
        Modeling Undersampling Case
        """
        print("------------------UNDERSAMPLING------------------")
        x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.2, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=20, max_depth=5)
        model.fit(x_under_train, y_under_train)
        scores = cross_val_score(model, x_under_train, y_under_train, cv=kf, scoring='accuracy')
        pred = model.predict(x_under_test)
        pred_prob = model.predict_proba(x_under_test)
        roc_auc = roc_auc_score(y_under_test, pred_prob[:,1])
        accuracy = accuracy_score(y_under_test, pred)
        print(classification_report(y_under_test, pred))
        print(f'ROC-AUC Score: {roc_auc}')
        print(f'Accuracy: {accuracy}')
        print(f"Scores for each fold: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        print(f"Standard deviation of accuracy: {scores.std():.4f}")
        X_shap = x_under_train
    elif data_preprocessing == "oversampling":
        """
        Modeling Oversampling Case
        """
        print("------------------OVERSAMPLING------------------")
        x_over_train, x_over_test, y_over_train, y_over_test = train_test_split(x_over, y_over, test_size=0.2, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=20, max_depth=5)
        model.fit(x_over_train, y_over_train)
        scores = cross_val_score(model, x_over_train, y_over_train, cv=kf, scoring='accuracy')
        pred = model.predict(x_over_test)
        pred_prob = model.predict_proba(x_over_test)
        roc_auc = roc_auc_score(y_over_test, pred_prob[:,1])
        accuracy = accuracy_score(y_over_test, pred)
        print(classification_report(y_over_test, pred))
        print(f'ROC-AUC Score: {roc_auc}')
        print(f'Accuracy: {accuracy}')
        print(f"Scores for each fold: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        print(f"Standard deviation of accuracy: {scores.std():.4f}")
        X_shap = x_over_train
    elif data_preprocessing == "smote":
        """
        Modeling SMOTE Case
        """
        print("------------------SMOTE------------------")
        x_smote_train, x_smote_test, y_smote_train, y_smote_test = train_test_split(x_smote, y_smote, test_size = 0.2, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=20, max_depth=5)
        model.fit(x_smote_train, y_smote_train)
        scores = cross_val_score(model, x_smote_train, y_smote_train, cv=kf, scoring='accuracy')
        pred = model.predict(x_smote_test)
        pred_prob = model.predict_proba(x_smote_test)
        roc_auc = roc_auc_score(y_smote_test, pred_prob[:,1])
        accuracy = accuracy_score(y_smote_test, pred)
        print(classification_report(y_smote_test, pred))
        print(f'ROC-AUC Score: {roc_auc}')
        print(f'Accuracy: {accuracy}')
        print(f"Scores for each fold: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f}")
        print(f"Standard deviation of accuracy: {scores.std():.4f}")
        X_shap = x_smote_train



    if send_model:
        feature_names = ["0_Rainfall_norm", "1_precipitation", "2_NDVI", "3_EVI", "4_ground_temp", "5_edge_distance", "6_elevation", "7_slope", "8_deforestation_density"] # <-- must match your band names in EE
        trees = ml.rf_to_strings(model, feature_names)

        geemap.ee_initialize()
        asset_id = "users/jtjayesh98/" + data_preprocessing + "_rf_" + str(start_year) + "_" + str(end_year) + "_plus_model"
        ml.export_trees_to_fc(trees, asset_id=asset_id, description="rf_export")
        classifier = ml.strings_to_classifier(trees)
        classifier.getInfo()
    background = shap.utils.sample(X_shap, 200, random_state=0)

    explainer = shap.TreeExplainer(
        model,
        data=background,                # "background"/reference distribution
        feature_names=X.columns,
        model_output="probability"      # for classifiers: "raw", "probability" or "log_loss"
    )

    sv = explainer(X_shap)

    if sv.values.ndim == 3:
        class_idx = 1
        vals = np.abs(sv.values[:, :, class_idx]).mean(axis=0)
    else:
        vals = np.abs(sv.values).mean(axis=0)

    global_importance = pd.Series(vals, index=X.columns).sort_values(ascending=False)
    print(global_importance.head(20))
    shap.plots.beeswarm(sv[..., 1], max_display=20)

    shap.plots.bar(sv[..., 1], max_display=20)
