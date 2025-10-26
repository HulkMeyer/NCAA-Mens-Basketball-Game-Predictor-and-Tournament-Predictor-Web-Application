from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from Bracket.routes import register_routes

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

register_routes(app)

