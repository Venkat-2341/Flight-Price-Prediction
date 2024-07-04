import pandas as pd
import joblib
from flask import Flask, render_template, request
from datetime import datetime
from forms2 import InputForm2

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

# Load the serialized pipeline containing both preprocessor and model
pipeline = joblib.load("pipeline2.joblib")

def calculate_duration(dep_time, arrival_time):
    dep_time_map = {
        "12 midnight to 6am": 0,
        "6am to 12 noon": 6,
        "12 noon to 6pm": 12,
        "6pm to 12 midnight": 18
    }
    dep_hour = dep_time_map[dep_time]
    arrival_hour = dep_time_map[arrival_time]
    duration = ((arrival_hour + 6 - dep_hour) % 24) * 60
    return duration

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route('/about')
def about():
    return render_template("about.html", title="About")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = InputForm2()
    if form.validate_on_submit():
        # Convert date_of_journey to datetime
        date_of_journey = pd.to_datetime(form.date_of_journey.data)

        # Calculate duration
        duration = calculate_duration(form.dep_time.data, form.arrival_time.data)
        
        # Determine if the journey is on a weekend
        is_weekend = int(date_of_journey.weekday() >= 5)

        # Prepare input data similar to the structure of your training data
        x_new = pd.DataFrame({
            'airline': [form.airline.data],
            'source': [form.source.data],
            'destination': [form.destination.data],
            'duration': [duration],
            'total_stops': [form.total_stops.data],
            'dep_time': [form.dep_time.data],
            'arrival_time': [form.arrival_time.data],
            'month': [date_of_journey.month],
            'day': [date_of_journey.day],
            'is_weekend': [is_weekend]
        })

        # Transform the input data using the preprocessor from the pipeline
        x_new_transformed = pipeline['preprocessor'].transform(x_new)

        # Make predictions using the model from the pipeline
        prediction = pipeline['model'].predict(x_new_transformed)[0]
        message = f"The predicted price is {prediction:,.0f} INR!"
    else:
        message = "Please provide valid input details!"

    return render_template("predict2.html", title="Predict", form=form, output=message)

if __name__ == "__main__":
    app.run(debug=True)
