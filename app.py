import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model
with open('model/he_model_rfe.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
    'age': 30,
    'sex': 1,
    'steroid': 0,
    'antivirals': 0,
    'fatigue': 0,
    'malaise': 0,
    'anorexia': 0,
    'liver_big': 0,
    'liver_firm': 0,
    'spiders': 0,
    'ascites': 0,
    'varices': 0,
    'bilirubin': 1,
    'alk_phosphate': 85,
    'sgot': 18,
    'albumin': 4,
    'histology': 0
}
        # user_input = {
        #     'age': float(request.form['age']),
        #     'sex': int(request.form['sex']),
        #     'steroid': int(request.form['steroid']), 
        #     'antivirals': int(request.form['antivirals']), 
        #     'fatigue': int(request.form['fatigue']),
        #     'malaise': int(request.form['malaise']), 
        #     'anorexia': int(request.form['anorexia']),
        #     'liver_big': int(request.form['liver_big']),	
        #     'liver_firm': int(request.form['liver_firm']),
        #     'spiders': int(request.form['spiders']),	 
        #     'ascites': int(request.form['ascites']),
        #     'varices': int(request.form['varices']), 
        #     'bilirubin': float(request.form['bilirubin']),
        #     'alk_phosphate': float(request.form['alk_phosphate']),
        #     'sgot': float(request.form['sgot']), 
        #     'albumin': float(request.form['albumin']),
        #     'histology': int(request.form['histology'])
        # }
        prediction = random_forest_model.predict([list(user_input.values())])
        # prediction = random_forest_model.predict([[user_input[key] for key in user_input]])
        if prediction[0] == 0:
            prediction_result = "Positive"
        else:
            prediction_result = "Negative"

        return render_template('index.html', prediction=prediction_result)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
