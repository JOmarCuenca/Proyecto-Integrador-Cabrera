#!/usr/bin/env python
from os import makedirs
from os.path import isdir, join
from flask import Flask, flash, request, redirect, render_template, Response
from werkzeug.utils import secure_filename
from camera import ModifiedCamera
from videoManager.breaker import cleanAndBreak

UPLOAD_FOLDER = './uploadedAssets'
ALLOWED_EXTENSIONS = {"mp4", "avi", "mkv", "webm"}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_camera = None


@app.route('/')
def index():
    """Video streaming home page."""
    if(current_camera is None):
        return render_template("empty.html")
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global current_camera
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            parentDir = app.config["UPLOAD_FOLDER"]
            if(not isdir(parentDir)):
                makedirs(parentDir)
            file.save(join(parentDir, filename))
            current_camera = ModifiedCamera(cleanAndBreak(parentDir+"/"+filename))
            return redirect("/")
    return render_template("uploadFile.html")

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(current_camera), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
