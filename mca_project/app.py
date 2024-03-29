import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# create flask app
app = Flask(__name__)
df = pd.read_csv('BCPD.csv')
x = df[["texture_mean", "area_mean", "concavity_mean", "area_se", "concavity_se",'fractal_dimension_se',
        "smoothness_worst", "concavity_worst", "symmetry_worst","fractal_dimension_worst"]]
y = df[["diagnosis"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# load pickle model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("prediction_form.html")



@app.route("/result", methods=["GET", "POST"])
def predict():
    l = []
    form_value = l
    if request.method == "POST":
        print(request.values)
    imd = request.form
    imd.to_dict(flat=False)
    print(imd)
    for k,v in imd.items():
        form_value.append(v)
    print(type(request.form))

    print(request.form)
    float_features = [float(x) for x in form_value]
    features = np.array([np.array(float_features)])
    sc=StandardScaler()
    Fit= sc.fit(x_train)
    features=Fit.transform(features)
    prediction = model.predict(features)
    if prediction[0] == 1:
        print("Malignant")
        return render_template("result.html", prediction_text=" MALIGNANT Cancer")
    else:
        print("Benign")
        return render_template("result.html", prediction_text=" BENIGN Cancer")

    # Python program to define a function to compute accuracy score of model's predicted class

    # Defining a function which takes true values of the sample and values predicted by the model
'''
def accuracy():
    classifier = KNeighborsClassifier()

    y_pred = classifier.predict(x_test)
    av = accuracy_score(y_test, y_pred)
    return render_template('result.html', av=av)
'''


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False,port=8080)
    ''''
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
        
    app.run(HOST, PORT)
    '''



