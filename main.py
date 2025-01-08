import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from sklearn.inspection import permutation_importance

def main():
    # Loading the data
    df = pd.read_csv('/Users/quentinvillet/kaggle_project/data/heart.csv')
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum().sort_values(ascending=False))
    print(df.isnull().sum().sort_values(ascending=False) / len(df))
    print(df.duplicated().sum())

    # Checking for outliers and understanding distribution of each feature
    numerical_data = df.select_dtypes(include=['int64', 'float64'])
    for feature in numerical_data:
        plt.figure()
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.show()

    print(df['HeartDisease'].value_counts() / len(df['HeartDisease']))

    # Checking for correlation between features
    corr_matrix = numerical_data.corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()

    print(df["Age"].describe())
    df["AgeGroup"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 70, 80], labels=["20-30", "30-40", "40-50", "50-60", "60-70", "70-80"])
    sns.countplot(data=df, x="AgeGroup", hue="HeartDisease")
    plt.title("Heart Disease by Age Group")
    plt.show()

    sns.countplot(data=df, x="ChestPainType", hue="HeartDisease")
    plt.title("Heart Disease by ChestPainType")
    plt.show()

    sns.countplot(data=df, x="ST_Slope", hue="HeartDisease")
    plt.title("Heart Disease by ST_Slope")
    plt.show()

    sns.countplot(data=df, x="RestingECG", hue="HeartDisease")
    plt.title("Heart Disease by RestingECG")
    plt.show()

    print(df.info())

    # Encoding categorical features
    Categorical_ordinal = df.select_dtypes(include='object')
    label = LabelEncoder()
    for col in Categorical_ordinal:
        df[col] = label.fit_transform(df[col])

    print(df)

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    # Picking the features (X) and the target (Y)
    X = df.drop(['HeartDisease', 'AgeGroup'], axis=1)
    y = df['HeartDisease']

    # Scaling Our data
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled = pd.concat([X_scaled_df, df[['HeartDisease', 'AgeGroup']]], axis=1)
    print(df_scaled)

    # Create my train and test groups
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # Modeling: first with Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred_logit = model.predict(X_test)
    accuracy_logit = accuracy_score(y_test, y_pred_logit)
    precision = precision_score(y_test, y_pred_logit)
    recall = recall_score(y_test, y_pred_logit)
    f1_logit = f1_score(y_test, y_pred_logit)

    results1 = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy_logit, precision, recall, f1_logit]
    })
    print(results1)

    cm = confusion_matrix(y_test, y_pred_logit)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys")
    plt.show()

    # Modeling: next with RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_RF = model.predict(X_test)
    accuracy_RF = accuracy_score(y_test, y_pred_RF)
    precision = precision_score(y_test, y_pred_RF)
    recall = recall_score(y_test, y_pred_RF)
    f1_RF = f1_score(y_test, y_pred_RF)

    results2 = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy_RF, precision, recall, f1_RF]
    })
    print(results2)

    cm = confusion_matrix(y_test, y_pred_RF)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.show()

    # Modeling: lastly with SVM
    svm = SVC(random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision = precision_score(y_test, y_pred_svm)
    recall = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)

    results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy_svm, precision, recall, f1_svm]
    })
    print(results)

    cm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    plt.show()

    # Checking Feature Permutation
    df_scaled
    X = df.drop(columns=['HeartDisease', 'AgeGroup'])
    y = df['HeartDisease']

    # Fit model
    log_model = LogisticRegression().fit(X, y)

    # Perform the permutation
    permutation_score = permutation_importance(log_model, X, y, n_repeats=10)

    # Unstack results showing the decrease in performance after shuffling features
    importance_df = pd.DataFrame(np.vstack((X.columns,
                                            permutation_score.importances_mean)).T)
    importance_df.columns = ['feature', 'score decrease']

    # Show the important features
    print(importance_df.sort_values(by="score decrease", ascending=False))

    final_result = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
        'Accuracy': [accuracy_logit, accuracy_RF, accuracy_svm],
        'F1 Score': [f1_logit, f1_RF, f1_svm]
    })
    final_result.set_index('Model', inplace=True)
    print(final_result)

if __name__ == "__main__":
    main()
