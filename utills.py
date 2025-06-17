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