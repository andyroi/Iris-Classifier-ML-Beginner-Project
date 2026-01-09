import numpy as np #numpy for numerical operations
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for applying data plots into matplot
from sklearn.datasets import load_iris # load the iris data
from sklearn.model_selection import train_test_split #to split data
from sklearn.tree import DecisionTreeClassifier, plot_tree #decision tree model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #by the names - will evaluate the confidence

iris = load_iris()

X = iris.data # X capital due to matrix (2D array) - convetion
#X contains all the measurements: sepal length, sepal width, etc.

y = iris.target # y lowercase because vector is (1D array) 
# y will be the labels of 0=setosa, 1=versicolor, 2=virginica

print("\n1. Splitting Data into Train/Test Sets")
print("   - Training set: 70% of data")
print("   - Test set: 30% of data")
print("   - random_state=42 ensures reproducibility")

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, #Freatuers to split
    y, # labels to split
    test_size = 0.3, #0.3 means 30% for testing and 70% of training
    random_state = 42 #random seed, same number always gives same split
    )

#going to print the sizes
print(f"\n Training Samples: {X_train.shape[0]}") #.shape[0] is number of rows
print(f"\n Test Samples: {X_test.shape[0]}")

#the main juice for creating the Decision Tree model
print("\n2. Training Decision Tree Classifier")
print("   - max_depth=3 prevents overfitting")

dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42) #DecisionTreeClassifier is using a tree and lowering the "impurity" as it goes down the list
# the DecisionTreeclassifier() creates a new decision tree object (the initializer) 
# max_depth=3: tree can only be 3 levels deep (preventing meorizing data)
# random_state = 42: makes results reproducible 

#this is the part where we are training the model
dt_classifier.fit(X_train, y_train)
#.fit() is THE learning method - it learns the pattern from X_train and y_train data
#after this ,the model will knwo how to classify the iris flowers
print("   ✓ Model trained successfully!")
print("\n3. Making Predictions on Test Set")
y_pred = dt_classifier.predict(X_test) #will predict post .fit or learn
#.predict() uses the trained model to guess labels for X_test
# y_pred contains the model's prediction

# first 10 predictions vs actual values
print(f" First 10 predictions: {y_pred[:10]}") # [:10] means first 10 items
print(f" Actual values: {y_test[:10]}")

#calculating the accuracy of the model:
print("\n4. Model Accuracy")
accuracy = accuracy_score(y_test, y_pred) #STILL CONFUSED ON THIS PART
#accuracy_score() compares actual (y_test) to predictions (y_pred) and should return between 0 and 1 (0% to 100%)

print(f" Accuracy: {accuracy * 100:.2f}%")
# {accuracy * 100:.2f} formats the number:
# * 100 converts to percentage
# :.2f means 2 decimal places with f for float

print("\n5. Confusion Matrix") #Pretty much looking at diagnol means correct and if any numbers populate outside from diagnol then it's being misclassed
print(" Shows how many predictions were correct/incorrect for each class")
cm = confusion_matrix(y_test, y_pred)
# Confusion matrix is a table showing:
# Rows = actual classes
# Columns = predicted classes
# Diagonal = correct predictions
print(cm)

print("\n6. Detailed Classification Report")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
#classifcatiion_report( showing the precision, recall, f1-score for each class)
# target_names gives readable names instead of 0,1,2

#creatiing visuals now with the confusion matrix and decision tree structure
#first confusion matrix as heatmap

fig, axes = plt.subplots(1,2, figsize = (16,6))
# 1, 2 means 1 row, 2 columns
# figsize=(16, 6) means 16 inches wide, 6 inches tall

#heatmap 3x3 of confusion matrix
sns.heatmap(
    cm,
    annot = True, #show numbers in cell
    fmt='d', #format integers to d = decimal
    cmap='Blues', #Color blue scheme
    xticklabels=iris.target_names, #x axis names
    yticklabels=iris.target_names, #y axis names
    ax=axes[0] #first plot
)
axes[0].set_title('Confusion Matrix - Decision Tree')
axes[0].set_ylabel('Actual')  # Label for y-axis
axes[0].set_xlabel('Predicted')  # Label for x-axis
                    

