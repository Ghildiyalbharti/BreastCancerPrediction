from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
#data = load_breast_cancer()
data= pd.read_csv('BCPD.csv')
X = data[["texture_mean", "area_mean", "concavity_mean", "area_se", "concavity_se",'fractal_dimension_se',
        "smoothness_worst", "concavity_worst", "symmetry_worst","fractal_dimension_worst"]]
#X = data.data
Y = data[["diagnosis"]]
#y = data.target


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Create a Naive Bayes model
model = GaussianNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

color = 'white'
matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))