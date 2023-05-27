from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("payments.pkl",'rb'))
app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['GET','POST'])
def Home():
    return render_template('home.html')

@app.route('/pred')
def Predict():
    return render_template('predict.html')

@app.route("/result", methods=['GET','POST'])
def Result():
    x = [[x for x in request.form.values()]]
    x = np.array(x)
    print(x.shape)
    print(x)
    pred = model.predict(x)
    print(pred[0])

    return render_template('submit.html',prediction_text=str(pred))

if __name__ == "__main__":
    app.run(debug=False)
