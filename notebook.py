# %%
import pandas as pd

# Load the datasets
x_train = pd.read_csv('./X_train.csv')
y_train = pd.read_csv('./y_train.csv')
x_test  = pd.read_csv('./X_test.csv')

# Display the first few rows of each dataset
x_train_head, y_train_head, x_test_head = x_train.head(), y_train.head(), x_test.head()
x_train_head, y_train_head, x_test_head


# %%
# Merge x_train and y_train using the appropriate columns
train = x_train.merge(y_train, left_on="id", right_on="Unnamed: 0")
train


# %%
# Drop the redundant columns after merging
train.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True)
train


# %%
# Total number of rows in the dataset
total_rows = train.shape[0]
total_rows


# %%
# Check for missing values
missing_values = train.isnull().sum()
missing_values


# %%
# Check for duplicates excluding the 'id' column
duplicates_excluding_id = train.duplicated(subset=train.columns.difference(['id'])).sum()
duplicates_excluding_id


# %%
# Summary statistics for numerical columns
numerical_summary = x_train.describe()
numerical_summary


# %%
# Remove duplicate rows (excluding the 'id' column)
train = train.drop_duplicates(subset=train.columns.difference(['id']))
train


# %%
# Remove rows with missing values
train = train.dropna()
train


# %%
#Distribution of these numerical columns
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up the style for visualization
sns.set_style("whitegrid")

# Plotting the distribution for numerical columns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Numerical Columns', fontsize=16)

sns.histplot(train['JoiningYear'], ax=axes[0, 0], kde=True)
axes[0, 0].set_title('Distribution of JoiningYear')

sns.histplot(train['PaymentTier'], ax=axes[0, 1], kde=True)
axes[0, 1].set_title('Distribution of PaymentTier')

sns.histplot(train['Age'], ax=axes[1, 0], kde=True)
axes[1, 0].set_title('Distribution of Age')

sns.histplot(train['ExperienceInCurrentDomain'], ax=axes[1, 1], kde=True)
axes[1, 1].set_title('Distribution of ExperienceInCurrentDomain')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()


# %%
age_stats = train['Age'].describe()
age_stats


# %%
export_path = "./combined_train_cleaned.csv"
train.to_csv(export_path, index=False)

export_path


# %%
# Categorical columns to explore
categorical_columns = ['Education', 'City', 'Gender', 'EverBenched', 'Race']

# Plotting the distribution for categorical columns
fig, axes = plt.subplots(len(categorical_columns), 1, figsize=(10, 14))
fig.suptitle('Distribution of Categorical Columns', fontsize=16)

for i, col in enumerate(categorical_columns):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# %% [markdown]
# <b>Education:</b> The majority of employees possess a Bachelor's degree, with a smaller proportion holding a Master's degree.<br />
# <b>City:</b> Bangalore is the predominant city where most employees are located, followed by Pune and Mumbai.<br />
# <b>Gender:</b> There are more male employees compared to female employees in the dataset.<br />
# <b>EverBenched:</b> A significant majority of employees have not experienced being benched.<br />
# <b>Race:</b> Most employees are identified as "white," with "black" being the next most common racial category.

# %%
# Drop the 'id' column and compute the correlation matrix again
correlation_matrix_without_id = train.drop(columns=['id']).corr()

# Plotting the heatmap without 'id'
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_without_id, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix (excluding id)')
plt.show()


# %%
# Outlier detection - Visualizing box plots for numerical columns
numerical_columns = ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=train[col])
    plt.title(f'Box plot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# %%
# Exporting the current dataframe
path = ".\combined_train_modified_before_encoding.csv"
train.to_csv(path, index=False)


# %%
# Encode categorial columns
# Display unique values in the 'Education' column
unique_education_values = train['Education'].unique()
unique_education_values


# %%
#Suggested Encoding: Label Encoding
#Bachelors can be encoded as 1
#Masters can be encoded as 2
#PHD can be encoded as 3

# Define a mapping for the Education column for label encoding
education_mapping = {
    'Bachelors': 1,
    'Masters': 2,
    'PHD': 3
}

# Apply the mapping to the Education column
train['Education_encoded'] = train['Education'].map(education_mapping)

# Display the first few rows to check the encoding
train[['Education', 'Education_encoded']].head()


# %%
#City: Nominal (the cities don't have a specific order or hierarchy) -> One-Hot Encoding
# Display unique values in the 'City' column and their count
unique_city_values = train['City'].nunique()
unique_city_values


# %%
# Check unique city values in the modified dataset
unique_cities_after_encoding = train['City'].unique()
unique_cities_after_encoding


# %%
# One-hot encode the 'City' column
city_dummies = pd.get_dummies(train['City'], prefix='City')
city_dummies


# %%
# Append the one-hot encoded columns to the dataframe
train = pd.concat([train, city_dummies], axis=1)
train


# %%
# Gender: Nominal (male and female don't have an intrinsic order) -> binary encoding
# Display unique values in the 'Gender' column
unique_gender_values = train['Gender'].unique()
unique_gender_values


# %%
# Define a mapping for the Gender column for binary encoding
gender_mapping = {
    'Male': 0,
    'Female': 1
}

# Apply the mapping to the Gender column
train['Gender_encoded'] = train['Gender'].map(gender_mapping)

# Display the first few rows to check the encoding
train[['Gender', 'Gender_encoded']].head()


# %%
#EverBenched: Nominal (binary in nature, indicating whether an employee has been benched or not) -> binary encoding
# Display unique values in the 'EverBenched' column
unique_everbenched_values = train['EverBenched'].unique()
unique_everbenched_values


# %%
# Define a mapping for the EverBenched column for binary encoding
everbenched_mapping = {
    'No': 0,
    'Yes': 1
}

# Apply the mapping to the EverBenched column
train['EverBenched_encoded'] = train['EverBenched'].map(everbenched_mapping)

# Display the first few rows to check the encoding
train[['EverBenched', 'EverBenched_encoded']].head()


# %%
# Race: Nominal (different races don't have an intrinsic order) -> One-Hot Encoding
# Display unique values in the 'Race' column
unique_race_values = train['Race'].unique()
unique_race_values


# %%
# One-hot encode the 'Race' column
race_dummies = pd.get_dummies(train['Race'], prefix='Race')

# Append the one-hot encoded columns to the dataframe
train = pd.concat([train, race_dummies], axis=1)

# Display the first few rows to check the encoding
train[['Race'] + [col for col in race_dummies.columns]].head()


# %%
# Drop the original categorical columns
columns_to_drop = ['Education', 'City', 'Gender', 'EverBenched', 'Race']
train.drop(columns=columns_to_drop, inplace=True)

# Display the first few rows of the updated dataframe
train.head()


# %%
# Exporting the current dataframe after dropping original categorical columns
path_after_drop = "./combined_train_after_dropping_categoricals.csv"
train.to_csv(path_after_drop, index=False)


# %%
# List of continuous features for visual exploration
continuous_features = ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']

# Plotting box plots for each continuous feature against the label 'LeaveOrNot'
plt.figure(figsize=(15, 10))
for i, feature in enumerate(continuous_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='LeaveOrNot', y=feature, data=train)
    plt.title(f'Box plot of {feature} vs LeaveOrNot')
    plt.xlabel('LeaveOrNot')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# %%
# Calculate correlations of all features with the 'LeaveOrNot' label
correlations_with_label = train.corr()['LeaveOrNot'].sort_values()

# Display correlations
correlations_with_label


# %%
# Plotting the heatmap for correlations
plt.figure(figsize=(18, 14))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')
plt.title('Correlation Heatmap')
plt.show()


# %% [markdown]
# Features like Race_yellow, Race_white, and Race_red have very low correlations with LeaveOrNot. While they could be candidates for removal, we should be cautious about potential non-linear relationships or interactions

# %%
#Scale Features
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Define the features to scale, excluding the 'LeaveOrNot' and 'id' columns
features_to_scale = train.columns.drop(['LeaveOrNot', 'id'])

# Apply scaling to the features
train[features_to_scale] = scaler.fit_transform(train[features_to_scale])

# Display the first few rows of the scaled dataframe
train.head()


# %%
# Exporting the current dataframe after dropping original categorical columns
path_after_scale = "./combined_train_scaling.csv"
train.to_csv(path_after_scale, index=False)


# %% [markdown]
# <b>Modeling</b>

# %%
#splitting the data.

# Set random_state
rnd_state = 10

from sklearn.model_selection import train_test_split, StratifiedKFold

# Separate the features and the target variable
X = train.drop(columns=['LeaveOrNot', 'id'])
y = train['LeaveOrNot']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rnd_state, stratify=y)

# Confirm the shape of the training and testing data
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# kfold
kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=rnd_state)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb

# Defining the classifiers and their hyperparameters
classifiers = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=rnd_state),
        'params': {
            'C': [0.3, 0.5, 1],
            'penalty': ['l1', 'l2']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=rnd_state),
        'params': {
            'max_depth': [7, 8, 9],
            'min_samples_split': [14, 15, 16]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3]
        }
    },
    'Gaussian Naive Bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=rnd_state),
        'params': {
            'n_estimators': [100, 110],
            'max_depth': [15],
            'min_samples_split': [11, 12],
            'min_samples_leaf': [3, 4],
            'max_features': ['auto'],
            'bootstrap': [False]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=rnd_state),
        'params': {
            'n_estimators': [120, 130],
            'learning_rate': [0.03],
            'max_depth': [5],
            'subsample': [0.9],
            'min_samples_split': [3, 4, 5]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=rnd_state),
        'params': {
            'n_estimators': [50],
            'learning_rate': [1.2, 1.3],
            'algorithm' : ['SAMME', 'SAMME.R']
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [110, 120],
            'learning_rate': [0.04, 0.05],
            'max_depth': [4, 5],
            'gamma': [0.1],
            'subsample': [0.9, 1.0],
            'colsample_bytree': [0.8, 0.9]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(force_row_wise=True, random_state=rnd_state),
        'params': {
            'n_estimators': [70, 80, 90],
            'learning_rate': [0.3, 0.4],
            'max_depth': [2,3, 4]
        }
    }
}


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

# To store the best model and its metrics for each classifier
results = {}

# Function to compute metrics for the best model from GridSearchCV
def compute_metrics(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return f1, auc, precision, recall, accuracy


# Iterating over each classifier and performing Grid Search
for classifier_name, classifier_info in classifiers.items():
    # GridSearchCV
    grid_search = GridSearchCV(classifier_info['model'], classifier_info['params'],
                               cv=kfold, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Compute metrics for the best model
    f1, auc, precision, recall, accuracy = compute_metrics(grid_search.best_estimator_, X_test, y_test)

    # Store results
    results[classifier_name] = {
        'F1 Score': f1,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Best Parameters': grid_search.best_params_
    }


results_df = pd.DataFrame(results).T
results_df


# %%
for item in results_df.index:
    print(item,results_df.loc[item]['Best Parameters'])


# %%


# %%
#TODO: Balancing The Dataset
from sklearn.utils import resample

#Separate majority and minority classes
df_majority = df[df.LeaveOrNot==0]
df_minority = df[df.LeaveOrNot==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=,
                                 random_state=10)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled['LeaveOrNot'].value_counts()

#Visualizing the Geography
df_upsampled['LeaveOrNot'].value_counts().plot(kind = 'bar')


# %%
#TODO: feature selection ?


# %%
