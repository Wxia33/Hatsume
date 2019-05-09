import cv.detect_game_shift as detect_shift
import os
from flask import Flask, request, redirect, url_for
from flask import render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './user_uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            if str(request.form['analysis_type']) == 'Split Frames':
                return redirect(url_for('vid_analysis', filename=filename))
    return render_template('index.html')


@app.route('/analysis/<filename>', methods=['GET', 'POST'])
def vid_analysis(filename):
    if request.method == 'GET':
        frame_change = detect_shift.game_change_detect('./user_uploads/' + filename)
    return redirect(url_for('results', frame_change=frame_change))


@app.route('/results/<frame_change>', methods=['GET', 'POST'])
def results(frame_change):
    return render_template('index.html', frame_change=frame_change)

if __name__ == "__main__":
    app.run()
