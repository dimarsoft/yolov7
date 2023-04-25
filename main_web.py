import base64
import os
from pathlib import Path

from flask import Flask, render_template, g, url_for, request, redirect

from configs import get_detections_path
from yolo_common.yolo_track_main import get_results_video_yolo, get_results_video_yolo_txt, download_test_video_1_85

app = Flask(__name__, template_folder='web/templates')

app.config['UPLOAD_FOLDER'] = './static/uploads'  # folder for uploaded files


# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('test_video_run', filename=filename))


@app.route('/download_videos', methods=['POST'])
def download_videos():
    output = app.config['UPLOAD_FOLDER']

    download_test_video_1_85(output)

    return render_template("index.html")
# return redirect(url_for('test_video_run', filename=filename))


@app.route('/upload_gdrive', methods=['POST'])
def upload_gdrive():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('test_video_run', filename=filename))


@app.route('/play/<filename>')
def play(filename):
    return render_template('play.html', filename=filename)


@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        pass
    else:
        return render_template("index.html")


@app.route('/about', methods=['POST', 'GET'])
def about():
    if request.method == 'POST':
        pass
    else:
        return render_template("about.html")


@app.route('/test', methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        pass
    else:
        return render_template("test.html", results=None)


@app.route('/test_video_run/<filename>')
def test_video_run(filename):
    # test_video = request.form.get("test_video")
    ff = str(Path(app.config['UPLOAD_FOLDER']) / filename)
    # ff = url_for('static', filename='uploads/' + filename)
    print(f"test video : {filename}")

    results = get_results_video_yolo_txt(ff, tracker_type="ocsort")

    devs = results["deviations"]
    fps = results["fps"]

    for item in devs:
        item["start_time"] = int(item["start_frame"] / fps)

    return render_template('test.html', test_video=filename, results=results)


if __name__ == '__main__':
    app.run(debug=True)
