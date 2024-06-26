import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def PredPasswordStrength(data_path):
    data = pd.read_csv(data_path)
    print("Dataset is:")
    print(data)
    data.head(30)
    print()
    data['strength'].unique()
    data.isna().sum()
    data[data['password'].isnull()]
    data.dropna(inplace = True)
    data.isnull().sum()
    plt.hist(data['strength'])
    plt.title("Password Strength")
    plt.show()

    def word_divide_char(inputs):
        character = []
        for i in inputs:
            character.append(i)
        return character

    word_divide_char('dragonoid@01')

    vectorizer = TfidfVectorizer(tokenizer = word_divide_char)
    
    x = data['password']
    y = data['strength']

    X = vectorizer.fit_transform(x)
    X.shape

    feature_names = list(vectorizer.vocabulary_.keys())
    print("Features are:")
    print(feature_names)
    print()

    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
    X_train.shape


    # LogisticRegression
    clf1 = LogisticRegression(random_state=0,multi_class='multinomial')
    clf1.fit(X_train,y_train)

    y_pred = clf1.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print()

    print("Result of Password Strength Using Logistic Regression Algorithm")
    print("Confusion Matrix:")
    print(cm)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1_score)
    print('Accuracy:', accuracy)
    print()

    # DecisionTreeClassifier
    clf2 = DecisionTreeClassifier(random_state=0)
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    print("Result of Password Strength Using Decision Tree Classifier Algorithm")
    print("Confusion Matrix:")
    print(cm)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1_score)
    print('Accuracy:', accuracy)
    print()

    # RandomForestClassifier
    clf3 = RandomForestClassifier(random_state=0)
    clf3.fit(X_train, y_train)
    y_pred = clf3.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    print("Result of Password Strength Using Random Forest Classifier Algorithm")
    print("Confusion Matrix:")
    print(cm)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1_score)
    print('Accuracy:', accuracy)
    print()

    dt=np.array(['eC3#yb4!z7cmbzA1'])
    pred=vectorizer.transform(dt)
    Ans1 = clf1.predict(pred)
    Ans2 = clf2.predict(pred)
    Ans3 = clf3.predict(pred)

    return Ans1,Ans2,Ans3

def main():
    print("---------------------------------------------------------------------------------------")
    print()
    print("--------------------------- Prediction of Password Strength ---------------------------")
    print()
    print("---------------------------------------------------------------------------------------")
    print("How to Verify the Password Strength??")
    print()
    print("If you get (Array output as '0') that means your password is Weak.")
    print("If you get (Array output as '1') that means your password is Strong.")
    print("If you get (Array output as '2') that means your password is Excellent/Very Strong.")
    print()
    print("---------------------------------------------------------------------------------------")

    Result1,Result2,Result3 = PredPasswordStrength('data_Small.csv')

    print("Prediction of your Password Strength Using Logistic Regression Algorithm: ",Result1)
    print("Prediction of your Password Strength Using Decision Tree Classifier Algorithm: ",Result2)
    print("Prediction of your Password Strength Using Random Forest Classifier Algorithm: ",Result3)
    print()
    print("------------------------ Thank you for using our Application --------------------------")
    print()

if __name__ == "__main__":
    main()

