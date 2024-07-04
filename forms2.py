import pandas as pd
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, SubmitField
from wtforms.fields.html5 import DateField as WTDateField
from wtforms.validators import DataRequired, NumberRange, ValidationError
from datetime import datetime, timedelta

# Load and preprocess your training data
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
train = pd.concat([train, val], axis=0)
train['date_of_journey'] = pd.to_datetime(train['date_of_journey'], format='%Y-%m-%d').dt.date
train['total_stops'] = train['total_stops'].astype(int)

def validate_date(form, field):
    if field.data <= datetime.today().date():
        raise ValidationError("Date of Journey must be from the next day onwards.")

def validate_source_destination(form, field):
    if form.source.data == form.destination.data:
        raise ValidationError("Source and destination cannot be the same.")
class InputForm2(FlaskForm):
    airline = SelectField(
        label="Airline",
        choices=train.airline.unique().tolist(),
        validators=[DataRequired()]
    )
    date_of_journey = WTDateField(
        label="Date of Journey",
        validators=[DataRequired(), validate_date]
    )
    source = SelectField(
        label="Source",
        choices=train.source.unique().tolist(),
        validators=[DataRequired(), validate_source_destination]
    )
    destination = SelectField(
        label="Destination",
        choices=train.destination.unique().tolist(),
        validators=[DataRequired(),validate_source_destination]
    )
    dep_time = SelectField(
        label="Departure Time",
        choices=[
            ("12 midnight to 6am", "12 midnight to 6am"),
            ("6am to 12 noon", "6am to 12 noon"),
            ("12 noon to 6pm", "12 noon to 6pm"),
            ("6pm to 12 midnight", "6pm to 12 midnight")
        ],
        validators=[DataRequired()]
    )
    arrival_time = SelectField(
        label="Arrival Time",
        choices=[
            ("12 midnight to 6am", "12 midnight to 6am"),
            ("6am to 12 noon", "6am to 12 noon"),
            ("12 noon to 6pm", "12 noon to 6pm"),
            ("6pm to 12 midnight", "6pm to 12 midnight")
        ],
        validators=[DataRequired()]
    )
    total_stops = IntegerField(
        label="Total Stops",
        validators=[DataRequired(), NumberRange(min=0)]
    )
    submit = SubmitField("Predict")
