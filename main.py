# To read our model
import pickle
import flask
from flask import Flask, jsonify
# to get the predicted value from the model
from model_files.ml_model import predict_mpg


app = Flask('mpg_prediction') # name of the app we can give any name

@app.route('/', methods=['POST']) # if we make a POST requst with some data then this method will be triggerd
def predict():
    # First we have to capture the user data for which user want the prediction
    vehicle_config = flask.request.get_json()

    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_mpg(vehicle_config, model)

    response = {
        'mpg_predictions': list(predictions)
    }

    # jsonify will convert our dictionary into a json type response
    return jsonify(response)




# This is just testing
'''
@app.route('/', methods=['GET'])
def ping():
    return 'Pinging model application !!'
'''

if __name__ == '__main__':
    app.run(debug=True) # , host='0.0.0.0', port=9696
    # debug : as soon as some changes occure in the file four server should start again
    # host is 0.0.0.0 so that we make our application public & can access from anywhere