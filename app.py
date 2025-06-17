import os, uuid, shutil, tempfile, threading, time, json
from flask import (
    Flask, render_template, request,
    send_from_directory, flash, redirect, url_for, Response
)
from pytube import YouTube
from utils import make_tab_pdf

app = Flask(__name__)
app.config["SECRET_KEY"] = "replace-with-your-secret"
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 진행률 저장소 { job_id: {"total":int, "current":int, "done":bool, "filename":str} }
progress = {}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    job_id = uuid.uuid4().hex
    tmpdir = tempfile.mkdtemp()
    yturl = request.form.get("yturl", "").strip()
    upload = request.files.get("videofile")

    # 1) YouTube 다운로드 or 2) 파일 업로드
    if yturl:
        try:
            yt = YouTube(yturl)
            stream = yt.streams.filter(progressive=True, file_extension="mp4")\
                               .order_by("resolution").desc().first()
            video_path = stream.download(output_path=tmpdir, filename="video.mp4")
        except Exception:
            flash("유튜브 다운로드에 실패했습니다.")
            shutil.rmtree(tmpdir)
            return redirect(url_for("index"))
    elif upload:
        video_path = os.path.join(tmpdir, "upload.mp4")
        upload.save(video_path)
    else:
        flash("유튜브 링크 또는 비디오 파일을 입력해 주세요.")
        shutil.rmtree(tmpdir)
        return redirect(url_for("index"))

    # 첫 프레임 썸네일 생성
    import cv2
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        flash("영상 읽기에 실패했습니다.")
        shutil.rmtree(tmpdir)
        return redirect(url_for("index"))

    thumb_name = f"thumb_{job_id}.jpg"
    thumb_path = os.path.join(app.config["UPLOAD_FOLDER"], thumb_name)
    cv2.imwrite(thumb_path, frame)

    # 진행률 초기화
    total = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    progress[job_id] = {"total": total, "current": 0, "done": False, "filename": ""}

    return render_template(
        "index.html",
        thumb=thumb_name,
        job_id=job_id,
        video_path=video_path
    )

@app.route("/process", methods=["POST"])
def process():
    job_id = request.form["job_id"]
    roi = [int(request.form[k]) for k in ("x","y","w","h")]
    rows = int(request.form.get("rows", 6))
    cols = int(request.form.get("cols", 1))
    video_path = request.form["video_path"]

    def task():
        def cb(frame_idx):
            progress[job_id]["current"] = frame_idx
        fname = make_tab_pdf(
            video_path,
            roi,
            rows=rows,
            cols=cols,
            sample_step=1,
            ssim_thresh=0.93,
            out_dir=app.config["UPLOAD_FOLDER"],
            progress_cb=cb
        )
        progress[job_id].update({"done": True, "filename": fname})
        # tmpdir 정리
        shutil.rmtree(os.path.dirname(video_path), ignore_errors=True)

    threading.Thread(target=task).start()
    return render_template("processing.html", job_id=job_id)

@app.route("/progress/<job_id>")
def progress_stream(job_id):
    def generate():
        while True:
            data = progress.get(job_id, {})
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("done"):
                break
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)