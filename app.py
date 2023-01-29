from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)
filename = 'processor.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    tdp = request.form['tdp']
    die_size = request.form['die_size']
    transistors = request.form['transistors']
    pred = model.predict(np.array([[tdp,die_size,transistors]]))
    print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run