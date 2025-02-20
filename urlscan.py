from flask import Flask, render_template, request, redirect
import jinja2
import sys
import numpy as np
import tld
from tld import get_tld
from urllib.parse import urlparse
import os
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__, static_folder='templates')
app.jinja_env.filters["zip"] = zip
app.debug = True

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

def build_conv1D_model():
    n_timesteps = 38
    n_features = 1
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_timesteps, n_features), name="Input"))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))
    model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='sigmoid', name="Dense_1"))
    model.add(keras.layers.Dense(n_features, activation='sigmoid', name="Dense_2"))

    model.compile(loss='mse', metrics=['accuracy'])
    return model

def analyzeUrl(url):
    try:
        qty_slash_url = url.count('/')
        length_url = len(url)

        url_path = urlparse(url).path
        url_directory = os.path.dirname(url_path)

        if url_directory=='':
            qty_dot_directory=-1
            qty_hyphen_directory=-1
            qty_underline_directory=-1
            qty_slash_directory=-1
            qty_questionmark_directory=-1
            qty_equal_directory=-1
            qty_at_directory=-1
            qty_and_directory=-1
            qty_exclamation_directory=-1
            qty_space_directory=-1
            qty_tilde_directory=-1
            qty_comma_directory=-1
            qty_plus_directory=-1
            qty_asterisk_directory=-1
            qty_hashtag_directory=-1
            qty_dollar_directory=-1
            qty_percent_directory=-1
            directory_length=-1
        else:
            qty_dot_directory=url_directory.count('.')
            qty_hyphen_directory=url_directory.count('-')
            qty_underline_directory=url_directory.count('_')
            qty_slash_directory=url_directory.count('/')
            qty_questionmark_directory=url_directory.count('?')
            qty_equal_directory=url_directory.count('=')
            qty_at_directory=url_directory.count('@')
            qty_and_directory=url_directory.count('&')
            qty_exclamation_directory=url_directory.count('!')
            qty_space_directory=url_directory.count(' ')
            qty_tilde_directory=url_directory.count('"')
            qty_comma_directory=url_directory.count(',')
            qty_plus_directory=url_directory.count('+')
            qty_asterisk_directory=url_directory.count('*')
            qty_hashtag_directory=url_directory.count('#')
            qty_dollar_directory=url_directory.count('$')
            qty_percent_directory=url_directory.count('%')
            directory_length=len(url_directory)
        
        url_file = os.path.basename(url_path)
        
        if url_file=='':
            qty_dot_file=-1
            qty_hyphen_file=-1
            qty_underline_file=-1
            qty_slash_file=-1
            qty_questionmark_file=-1
            qty_equal_file=-1
            qty_at_file=-1
            qty_and_file=-1
            qty_exclamation_file=-1
            qty_space_file=-1
            qty_tilde_file=-1
            qty_comma_file=-1
            qty_plus_file=-1
            qty_asterisk_file=-1
            qty_hashtag_file=-1
            qty_dollar_file=-1
            qty_percent_file=-1
            file_length=-1
        else:
            qty_dot_file=url_file.count('.')
            qty_hyphen_file=url_file.count('-')
            qty_underline_file=url_file.count('_')
            qty_slash_file=url_file.count('/')
            qty_questionmark_file=url_file.count('?')
            qty_equal_file=url_file.count('=')
            qty_at_file=url_file.count('@')
            qty_and_file=url_file.count('&')
            qty_exclamation_file=url_file.count('!')
            qty_space_file=url_file.count(' ')
            qty_tilde_file=url_file.count('"')
            qty_comma_file=url_file.count(',')
            qty_plus_file=url_file.count('+')
            qty_asterisk_file=url_file.count('*')
            qty_hashtag_file=url_file.count('#')
            qty_dollar_file=url_file.count('$')
            qty_percent_file=url_file.count('%')
            file_length=len(url_file)
        

        extract_url=[qty_slash_url,
                    length_url,
                    qty_dot_directory,
                    qty_hyphen_directory,
                    qty_underline_directory,
                    qty_slash_directory,
                    qty_questionmark_directory,
                    qty_equal_directory,
                    qty_at_directory,
                    qty_and_directory,
                    qty_exclamation_directory,
                    qty_space_directory,
                    qty_tilde_directory,
                    qty_comma_directory,
                    qty_plus_directory,
                    qty_asterisk_directory,
                    qty_hashtag_directory,
                    qty_dollar_directory,
                    qty_percent_directory,
                    directory_length,
                    qty_dot_file,
                    qty_hyphen_file,
                    qty_underline_file,
                    qty_slash_file,
                    qty_questionmark_file,
                    qty_equal_file,
                    qty_at_file,
                    qty_and_file,
                    qty_exclamation_file,
                    qty_space_file,
                    qty_tilde_file,
                    qty_comma_file,
                    qty_plus_file,
                    qty_asterisk_file,
                    qty_hashtag_file,
                    qty_dollar_file,
                    qty_percent_file,
                    file_length]
        

        extract_url = np.array(extract_url)
        print(extract_url)
        extract_url_reshaped = extract_url.reshape(1,38,1)
        print(extract_url_reshaped)

        # Load the model and weights
        new_model_conv1D = build_conv1D_model()
        new_model_conv1D.load_weights('model_conv1D_spearman.h5')

        # Use the LOADED model for predictions!
        predictions = new_model_conv1D.predict(extract_url_reshaped).flatten()
        predictions = ['Legitimate' if item < 0.73 else 'Phishing' for item in predictions]
        predictions = listToString(predictions)

    except tld.exceptions.TldBadUrl:
        predictions = "Url yang dimasukan kurang lengkap harus dengan http:// atau https://"

    except Exception as e:
        predictions = "Program Error " + str(e)

    return predictions

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/data', methods=['POST', 'GET'])
def data():
    if request.method == 'POST':
        form_data = request.form.get('Url')
        hasil = analyzeUrl(str(form_data))
        dataJson = {
            "form_data": [form_data],
            "hasil": [hasil],
        }
        return render_template('data.html', data=dataJson, zip=zip)
    else:
        return redirect('/')

if __name__ == '__main__':
    app.run(host='localhost', port=5000)