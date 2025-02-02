from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for a home page
@app.route('/')
def index():
    name = "Flask User, Madhav"
    return render_template('index.html', name=name)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        input_data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        
        input_df = input_data.get_input_data_as_data_frame()
        print(input_df)
        
        predict_pipeline = PredictPipeline()
        math_score_results = predict_pipeline.predict(input_df)
        return render_template("home.html", results=math_score_results[0])

# Run the application if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
