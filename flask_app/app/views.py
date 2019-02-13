import os
from flask import render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from app import app
from app.DogBreedNet import dog_breed_net

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['POST'])
def upload_file():
  f = request.files['file']
  output_text=dog_breed_net(f)
  return render_template('index.html', output_text=output_text)

@app.route('/about')
def about():
  return render_template('about.html')
