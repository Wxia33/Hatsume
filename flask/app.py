import cv.detect_game_shift as detect_shift
from flask import Flask
from flask import render_template


app = Flask(__name__)


@app.route('/')
def index():
    user = {'username': 'Will'}
    return render_template('index.html', title='Home', user=user)


@app.route('/analyze')
def analysis():
    vid_file = ""
    shift_frames = detect_shift.game_change_detect(vid_file)
    user = {'username': 'Will'}
    return render_template('analysis.html', title='Home', user=user, shift_frames=shift_frames)

