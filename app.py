from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('df.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    area = request.form.get('area')
    input_query = np.array([area])
    result = model.predict(input_query)[0]
    return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run(debug=True)