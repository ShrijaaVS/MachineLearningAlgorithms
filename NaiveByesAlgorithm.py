import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('/Users/shrijaa/Desktop/Python_ML/Datasets/User_data.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting the Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Finding the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy Score: {:.4f}'.format(accuracy))

# Checking the train-set accuracy
y_pred_train = classifier.predict(x_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print('Training-set Accuracy Score: {:.4f}'.format(train_accuracy))

# Scores on the training set and test set
print('Training set score: {:.4f}'.format(classifier.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(classifier.score(x_test, y_test)))

# Comparing with null accuracy - accuracy achieved by always predicting the most frequent class
unique_classes, class_counts = np.unique(y_test, return_counts=True)
most_frequent_class = unique_classes[np.argmax(class_counts)]
null_accuracy = np.max(class_counts) / len(y_test)
print('Null Accuracy: {:.4f}'.format(null_accuracy))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)

# Classification Report
print('Classification Report:\n', classification_report(y_test, y_pred))

# Visualizing Confusion Matrix with seaborn heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive: 1', 'Actual Negative: 0'],
                         index=['Predicted Positive: 1', 'Predicted Negative: 0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.show()

# Classification Accuracy
classification_accuracy = accuracy_score(y_test, y_pred)
print('Classification accuracy: {:.4f}'.format(classification_accuracy))

# Precision
precision = cm[0, 0] / np.sum(cm[:, 0])
print('Precision: {:.4f}'.format(precision))

# Recall
recall = cm[0, 0] / np.sum(cm[0, :])
print('Recall or Sensitivity or True Positive Rate: {:.4f}'.format(recall))

# False Positive Rate
false_positive_rate = cm[1, 0] / np.sum(cm[1, :])
print('False Positive Rate: {:.4f}'.format(false_positive_rate))

# Specificity
specificity = cm[1, 1] / np.sum(cm[1, :])
print('Specificity: {:.4f}'.format(specificity))

# f1-Score
f1_score = 2 * (precision * recall) / (precision + recall)
print('f1-Score: {:.4f}'.format(f1_score))

#ROC AUC Score => Single number summray of classifier performance(higher the value , better the classifer)
ROC_AUC = roc_auc_score(y_test, y_pred)
print('ROC AUC : {:.4f}'.format(ROC_AUC))

#Cross-validated ROC AUC
cross_validated_roc_auc=cross_val_score(classifier,x_train,y_train,cv=5,scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(cross_validated_roc_auc))
#10-Fold Cross Validation
scores = cross_val_score(classifier, x_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

# compute Average cross-validation score
print('Average cross-validation score: {:.4f}'.format(scores.mean()))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#visualising the training set results
x_set,y_set=x_train,y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
             alpha = 0.75, cmap = ListedColormap(('purple', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i , j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
    c=ListedColormap(('purple','green'))(i),label=j)
plt.title('Naive Bayes - Training Set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#visualising the test set results
x_set,y_Set=x_test,y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
             alpha = 0.75, cmap = ListedColormap(('purple', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i , j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
    c=ListedColormap(('purple','green'))(i),label=j)
plt.title('Naive Bayes - Testing Set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
