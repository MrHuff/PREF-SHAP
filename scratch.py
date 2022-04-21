import xgboost
import shap

# train XGBoost model
X, y = shap.datasets.adult()
model = xgboost.XGBClassifier().fit(X.iloc[:1000, :], y[:1000])
#
# # compute SHAP values
explainer = shap.Explainer(model, X.iloc[:1000, :])
shap_values_ref = explainer(X.iloc[:1000, :])
shap.summary_plot(shap_values_ref,X.iloc[:1000, :])