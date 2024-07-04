import pandas as pd
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, SubmitField
from wtforms.fields.html5 import DateField as WTDateField, TimeField as WTTimeField
from wtforms.validators import DataRequired, NumberRange
from sklearn.preprocessing import LabelEncoder

# Load and preprocess your training data
train = pd.read_csv("data/train.csv")
train = train.drop(columns="price")
train['date_of_journey'] = pd.to_datetime(train['date_of_journey'], format='%Y-%m-%d').dt.date
train['duration'] = train['duration'].astype(int)
train['total_stops'] = train['total_stops'].astype(int)

class InputForm(FlaskForm):
    airline = SelectField(
        label="Airline",
        choices=train.airline.unique().tolist(),
        validators=[DataRequired()]
    )
    date_of_journey = WTDateField(
        label="Date of Journey",
        validators=[DataRequired()]
    )
    source = SelectField(
        label="Source",
        choices=train.source.unique().tolist(),
        validators=[DataRequired()]
    )
    destination = SelectField(
        label="Destination",
        choices=train.destination.unique().tolist(),
        validators=[DataRequired()]
    )
    dep_time = WTTimeField(
        label="Departure Time",
        validators=[DataRequired()]
    )
    arrival_time = WTTimeField(
        label="Arrival Time",
        validators=[DataRequired()]
    )
    duration = IntegerField(
        label="Duration",
        validators=[DataRequired(), NumberRange(min=0)]
    )
    total_stops = IntegerField(
        label="Total Stops",
        validators=[DataRequired(), NumberRange(min=0)]
    )
    additional_info = SelectField(
        label="Additional Info",
        choices=train.additional_info.unique().tolist(),
        validators=[DataRequired()]
    )
    submit = SubmitField("Predict")
