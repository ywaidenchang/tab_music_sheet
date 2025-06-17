# tab_music_sheet
convert video music sheet to pdf music sheet


아래는 **Bootstrap 5** 기반으로 디자인을 개선하고, 유튜브 링크/영상 업로드, ROI 선택, 진행률 표시, PDF 다운로드까지 전 과정을 구현한 전체 코드입니다.

---

## 프로젝트 구조

```
tab_web/
├── app.py
├── utils.py
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── processing.html
│   └── result.html
└── static/
    ├── css/
    │   └── bootstrap.min.css
    ├── js/
    │   └── bootstrap.bundle.min.js
    └── uploads/           # 썸네일·다운로드된 PDF 임시 저장 폴더
```

---

## requirements.txt

```txt
Flask
pytube3
opencv-python
Pillow
scikit-image
reportlab
tqdm
```

---

## app.py

```python
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
```

---

## utils.py

```python
import cv2, os, uuid
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

def make_tab_pdf(
    mp4_path, roi,
    rows=6, cols=1,
    sample_step=1, ssim_thresh=0.95,
    out_dir="static/uploads",
    progress_cb=None
):
    x, y, w, h = map(int, roi)
    cap = cv2.VideoCapture(mp4_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fname = f"tab_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = os.path.join(out_dir, fname)

    # PDF 캔버스 세팅
    page_w, page_h = A4
    cell_w, cell_h = page_w/cols, page_h/rows
    canv = canvas.Canvas(pdf_path, pagesize=A4)

    prev_gray = None
    idx = 0

    for i in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_step != 0:
            if progress_cb: progress_cb(i)
            continue

        roi_img = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            score = ssim(gray, prev_gray)
            if score > ssim_thresh:
                if progress_cb: progress_cb(i)
                continue
        prev_gray = gray

        # PDF 그리드 배치
        row, col = divmod(idx, cols)
        if row == rows:
            canv.showPage()
            idx = 0
            row, col = 0, 0

        x0, y0 = col*cell_w, page_h-(row+1)*cell_h
        pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        iw, ih = pil.size
        ratio = min(cell_w/iw, cell_h/ih)
        iw, ih = iw*ratio, ih*ratio

        buf = BytesIO()
        pil.save(buf, format="JPEG")
        buf.seek(0)
        canv.drawImage(
            ImageReader(buf),
            x0 + (cell_w-iw)/2,
            y0 + (cell_h-ih)/2,
            width=iw,
            height=ih
        )
        buf.close()

        idx += 1
        if progress_cb:
            progress_cb(i)

    canv.showPage()
    canv.save()
    cap.release()
    return fname
```

---

## templates/index.html

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>베이스 TAB PDF 변환기</title>
  <link rel="stylesheet" href="/static/css/bootstrap.min.css">
  <style>
    body { padding-top: 2rem; }
    #canvas { border: 1px solid #dee2e6; cursor: crosshair; max-width: 100%; }
  </style>
</head>
<body class="bg-light">
  <div class="container">
    <h1 class="mb-4 text-center">🎸 TAB PDF 변환기</h1>

    <!-- 1. 입력 폼 -->
    <div class="card mb-4">
      <div class="card-body">
        <form method="post" action="/submit" enctype="multipart/form-data">
          <div class="mb-3">
            <label class="form-label">YouTube URL</label>
            <input type="url" name="yturl" class="form-control" placeholder="https://www.youtube.com/..." />
          </div>
          <div class="mb-3">
            <label class="form-label">비디오 파일 업로드</label>
            <input type="file" name="videofile" accept="video/*" class="form-control" />
          </div>
          <button type="submit" class="btn btn-primary w-100">다음 &gt;</button>
        </form>
      </div>
    </div>

    <!-- 2. 썸네일 + ROI 선택 -->
    {% if thumb %}
    <div class="card mb-4">
      <div class="card-body">
        <h5 class="card-title">악보 영역 선택</h5>
        <canvas id="canvas"></canvas>
        <form id="roiForm" method="post" action="/process">
          <input type="hidden" name="x" /><input type="hidden" name="y" />
          <input type="hidden" name="w" /><input type="hidden" name="h" />
          <input type="hidden" name="rows" value="6" />
          <input type="hidden" name="cols" value="1" />
          <input type="hidden" name="job_id" value="{{ job_id }}" />
          <input type="hidden" name="video_path" value="{{ video_path }}" />
          <button id="btnRoi" class="btn btn-success mt-3" disabled>PDF 생성 시작</button>
        </form>
      </div>
    </div>
    {% endif %}
  </div>

  <script src="/static/js/bootstrap.bundle.min.js"></script>
  {% if thumb %}
  <script>
    const img = new Image(), canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
    img.src = `/static/uploads/{{ thumb }}`;
    let start=null, rect=null;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    canvas.onmousedown = e => { start = {x:e.offsetX,y:e.offsetY}; };
    canvas.onmousemove = e => {
      if (!start) return;
      rect = {
        x: Math.min(start.x, e.offsetX),
        y: Math.min(start.y, e.offsetY),
        w: Math.abs(start.x - e.offsetX),
        h: Math.abs(start.y - e.offsetY)
      };
      ctx.drawImage(img, 0, 0);
      if (rect.w && rect.h) {
        ctx.strokeStyle = '#dc3545';
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      }
    };
    canvas.onmouseup = () => {
      if (rect) {
        ['x','y','w','h'].forEach(k => {
          document.querySelector(`input[name=${k}]`).value = rect[k];
        });
        document.getElementById('btnRoi').disabled = false;
      }
      start = null;
    };
  </script>
  {% endif %}
</body>
</html>
```

---

## templates/processing.html

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>PDF 생성 중...</title>
  <link rel="stylesheet" href="/static/css/bootstrap.min.css">
  <style>body{padding-top:4rem;}</style>
</head>
<body class="bg-light">
  <div class="container text-center">
    <h2>PDF를 생성 중입니다...</h2>
    <div class="progress mt-4">
      <div id="bar" class="progress-bar progress-bar-striped progress-bar-animated"
           role="progressbar" style="width: 0%;"></div>
    </div>
  </div>
  <script>
    const source = new EventSource("/progress/{{ job_id }}");
    source.onmessage = e => {
      const d = JSON.parse(e.data);
      if (d.total) {
        const pct = Math.floor(d.current / d.total * 100);
        document.getElementById("bar").style.width = pct + "%";
      }
      if (d.done) {
        source.close();
        window.location.href = `/download/${d.filename}`;
      }
    };
  </script>
</body>
</html>
```

---

## templates/result.html

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>완료</title>
  <link rel="stylesheet" href="/static/css/bootstrap.min.css">
</head>
<body class="bg-light">
  <div class="container text-center" style="padding-top:4rem;">
    <h2>✅ PDF 생성이 완료되었습니다!</h2>
    <p><a href="/download/{{ pdf }}" class="btn btn-success">다운로드</a></p>
  </div>
</body>
</html>
```

---

> **설치 & 실행**
>
> ```bash
> pip install -r requirements.txt
> python app.py
> ```
>
> 그 후 `http://localhost:5000` 에 접속하세요.
> Bootstrap 스타일로 깔끔하게, 유튜브 링크·파일 업로드 → ROI 지정 → 실시간 진행률 → PDF 다운로드까지 원클릭 체험이 가능합니다!
