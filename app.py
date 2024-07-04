import pandas as pd
import joblib
from flask import (
    Flask,
    render_template,
    request
)
from forms import InputForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

# Load the serialized pipeline containing both preprocessor and model
pipeline = joblib.load("pipeline.joblib")

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route('/about')
def about():
    return render_template("about.html",title="About")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        # Convert date_of_journey to datetime
        date_of_journey = pd.to_datetime(form.date_of_journey.data)

        # Prepare input data similar to the structure of your training data
        x_new = pd.DataFrame({
            'airline': [form.airline.data],
            'source': [form.source.data],
            'destination': [form.destination.data],
            'duration': [form.duration.data],
            'total_stops': [form.total_stops.data],
            'additional_info': [form.additional_info.data],
            'journey_day_of_week': [date_of_journey.dayofweek],
            'journey_day_of_month': [date_of_journey.day],
            'journey_month': [date_of_journey.month],
            'dep_hour': [form.dep_time.data.hour],
            'dep_minute': [form.dep_time.data.minute],
            'arrival_hour': [form.arrival_time.data.hour],
            'arrival_minute': [form.arrival_time.data.minute],
        })

        # Transform the input data using the preprocessor from the pipeline
        x_new_transformed = pipeline['preprocessor'].transform(x_new)

        # Make predictions using the model from the pipeline
        prediction = pipeline['model'].predict(x_new_transformed)[0]
        message = f"The predicted price is {prediction:,.0f} INR!"
    else:
        message = "Please provide valid input details!"

    return render_template("predict.html", title="Predict", form=form, output=message)

if __name__ == "__main__":
    app.run(debug=True)

