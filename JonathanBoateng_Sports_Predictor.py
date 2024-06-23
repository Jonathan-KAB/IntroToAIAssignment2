# %%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from category_encoders import TargetEncoder

# %%
def preprocess_data(file_path, n_features=1000):
    # Load the CSV file
    fifdb = pd.read_csv(file_path)
    
    print(f"Initial shape: {fifdb.shape}")
    
    # Dropping specified columns
    columns_to_drop = [
        'player_url', 'value_eur', 'wage_eur', 'player_face_url', 'club_flag_url', 'nation_logo_url', 
        'nation_flag_url', 'club_logo_url', 'club_team_id'
    ]
    new_fifdb = fifdb.drop(columns_to_drop, axis=1)
    
    print(f"Shape after dropping columns: {new_fifdb.shape}")
    
    # Extracting the 'overall' column as the target variable (y)
    y = new_fifdb['overall']
    
    # Dropping 'overall' from the features
    new_fifdb.drop('overall', axis=1, inplace=True)
    
    # Identify numeric and categorical columns
    numeric_features = new_fifdb.select_dtypes(include=[np.number]).columns
    categorical_features = new_fifdb.select_dtypes(include=['object']).columns
    
    print(f"Number of numeric features: {len(numeric_features)}")
    print(f"Number of categorical features: {len(categorical_features)}")
    
    # Handle high-cardinality categorical variables
    for col in categorical_features:
        if new_fifdb[col].nunique() > 100:  # Adjust this threshold as needed
            top_100 = new_fifdb[col].value_counts().nlargest(100).index
            new_fifdb[col] = new_fifdb[col].where(new_fifdb[col].isin(top_100), 'Other')
    
    # Defining pipelines for numeric and categorical data
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())
    ])
    
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", TargetEncoder())
    ])
    
    # Combining pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])
    
    # Feature selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), 
                               max_features=n_features, threshold=-np.inf)
    
    # Combine preprocessing and feature selection
    full_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("select", selector)
    ])
    
    # Fit and transform the data
    X = full_pipeline.fit_transform(new_fifdb, y)
    
    print(f"Final shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    return X, y, full_pipeline

# %%
X, y, pipeline = preprocess_data('players_21.csv', n_features=10)

# %%
print(X, y)

# %%
X.shape

# %%
y.shape

# %%
y.iloc[25:50]

# %% [markdown]
# ### Train-Test Split

# %%
from sklearn.model_selection import train_test_split

# %%
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# %%
dt = DecisionTreeClassifier(criterion='entropy')
knn = KNeighborsClassifier(n_neighbors=7)
sv = SVC(probability=True)
nb = GaussianNB()

# %% [markdown]
# ### Regression
# **Regressors used**
# * Multilinear Regression
# * Decision Tree

# %% [markdown]
# #### Multilinear Regression

# %%
from sklearn.linear_model import LinearRegression

# %%
l = LinearRegression()

# %%
# Learning association between X and Y
l.fit(Xtrain, Ytrain)

# %%
y_pred = l.predict(Xtest)

# %%
intercept = l.intercept_
coefficients = l.coef_

# %%
print(f"Intercept: {intercept}")
print(f"Coefficients: {coefficients}")

# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

# %%
print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

# %% [markdown]
# #### Decision Tree

# %%
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(max_depth = 15)
dtree.fit(Xtrain, Ytrain)

# Model testing
y_pred = dtree.predict(Xtest)

# %%
print(f"""
Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
R2 Score = {r2_score(y_pred,Ytest)}
""")

# %% [markdown]
# ### Ensemble Classification

# %% [markdown]
# **Classifers used**
# * Boosting

# %% [markdown]
# #### Boosting

# %%
import pickle as pkl
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve

# %%
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# %%
ada=AdaBoostClassifier(base_estimator=dt,n_estimators=100)
gb=GradientBoostingClassifier(n_estimators=100)

# %%
for model in (ada,gb):
 model.fit(Xtrain, Ytrain)
 pkl.dump(model, open('./' + model.__class__.__name__ + '.pkl', 'wb'))
 y_pred = model.predict(Xtest)
 print(model.__class__.__name__, confusion_matrix(Ytest, y_pred), classification_report(Ytest, y_pred))

# %% [markdown]
# ### Ensemble Modelling
# **Modelling techniques used:**
# * RandomForest

# %% [markdown]
# #### RandomForest

# %%
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(Xtrain, Ytrain)

# %%
y_pred = rf_model.predict(Xtest)
accuracy = accuracy_score(Ytest, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(Ytest, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(Ytest, y_pred))

# %%
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

print("Model saved as rf_model.pkl")

# %% [markdown]
# ### Hyperparameter tuning

# %%
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# %%
from sklearn.model_selection import RandomizedSearchCV

# Define base estimator
rf_base_estimator = RandomForestRegressor()

# Create ensemble model
ensemble = AdaBoostRegressor(base_estimator=rf_base_estimator)

# Define hyperparameter distributions
param_distributions = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'base_estimator__n_estimators': [100, 200, 300],
    'base_estimator__max_depth': [None, 10, 20]
}

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(ensemble, param_distributions, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit RandomizedSearchCV
random_search.fit(Xtrain, Ytrain)

# Make predictions
y_pred = random_search.predict(Xtest)

# Evaluate the model
mse = mean_squared_error(Ytest, y_pred)
r2 = r2_score(Ytest, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# %%
# Retrieve feature importances from the best estimator
importances = grid_search.best_estimator_.base_estimator.feature_importances_

# Create an array of feature names
feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])

# Create a DataFrame with feature names and importances
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Print top 20 important features
print(feature_importances.head(20))

# %%
# pip install streamlit


