import cv2, time, sqlite3, pathlib, enum
from collections import deque
from datetime import datetime

# ── STREAM URLs ───────────────────────────────────────────────────────────────
M3U8_HTTPS = "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"
M3U8_HTTP  = "http://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"

# ── YOUR ROIs (x, y, w, h) ────────────────────────────────────────────────────
ROI_BOT = (577, 773, 92, 107)   # bottom band
ROI_MID = (660, 499, 33, 101)   # mid-tower band
ROI_TOP = (507, 113, 47, 82)    # crest band

# ── DETECTION THRESHOLDS ──────────────────────────────────────────────────────
MOTION_ENTER = 1000     # pixel-count to say “train present”
MOTION_EXIT  = 200
CLASSIFY_MS  = 8000     # <8 s up-motion on tower ⇒ natural rollback
WAIT_TIMEOUT = 10       # give backward launch ≤10 s to re-appear

DB_PATH = pathlib.Path("events.db")

# ── STATE MACHINE ENUM ────────────────────────────────────────────────────────
class S(enum.Enum):
    IDLE = 0; ASC1 = 1; RBACK = 2; WAIT = 3; ASC3 = 4

# ── STREAM HELPERS ────────────────────────────────────────────────────────────
def open_stream() -> cv2.VideoCapture:
    """Force FFmpeg backend; try HTTPS first, then HTTP."""
    for url in (M3U8_HTTPS, M3U8_HTTP):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f"[stream] connected via {url.split(':')[0].upper()}")
            return cap
        cap.release()
    raise RuntimeError("Could not open HLS stream with FFmpeg backend")

def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS launches (
            id INTEGER PRIMARY KEY,
            ts_utc  TEXT,
            outcome TEXT  -- success | rollback | incomplete
        );
    """)

def log_event(conn, outcome):
    conn.execute(
        "INSERT INTO launches(ts_utc, outcome) VALUES (?,?)",
        (datetime.utcnow().isoformat(timespec="seconds"), outcome)
    )
    conn.commit()
    print(f"[event] {outcome} @ {datetime.now().strftime('%H:%M:%S')}")

def in_roi(cy, roi):
    return roi[1] <= cy <= roi[1] + roi[3]

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    cap   = open_stream()
    bg    = None                          # adaptive background for ROI_BOT
    state = S.IDLE
    hist  = deque(maxlen=2)               # store last two (cy, t) pairs
    t_wait = None

    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[stream] dropped – reconnecting in 2 s…")
            time.sleep(2)
            cap.release()
            cap = open_stream()
            continue

        # ── 1️⃣  Crop bottom ROI and measure motion ─────────────
        x,y,w,h = ROI_BOT
        roi = frame[y:y+h, x:x+w]

        if bg is None:
            bg = roi.astype("float32")
            continue

        diff    = cv2.absdiff(roi, bg.astype("uint8"))
        motion  = (diff > 25).sum()

        # crude vertical centroid in ROI_BOT
        gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binmask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        m      = cv2.moments(binmask)
        cy     = y + int(m['m01']/m['m00']) if m['m00'] else None

        # velocity (down=+, up=–) in pixels/frame
        if cy is not None:
            hist.append((cy, time.time()))
        v = (hist[-1][0] - hist[-2][0]) if len(hist) >= 2 else 0
        now = time.time()

        # ── 2️⃣  FINITE-STATE MACHINE ──────────────────────────
        if state == S.IDLE and motion > MOTION_ENTER and v < -5:
            state = S.ASC1

        elif state == S.ASC1 and v > 5 and cy and in_roi(cy, ROI_BOT):
            state = S.RBACK

        elif state == S.RBACK and motion < MOTION_EXIT:
            state, t_wait = S.WAIT, now

        elif state == S.WAIT and motion > MOTION_ENTER and v < -5:
            state = S.ASC3

        elif state == S.WAIT and now - t_wait > WAIT_TIMEOUT:
            log_event(conn, "incomplete")
            state = S.IDLE

        elif state == S.ASC3 and cy and in_roi(cy, ROI_TOP):
            log_event(conn, "success")
            state = S.IDLE

        elif (
            state == S.ASC3
            and v > 5
            and cy and cy > ROI_MID[1]
        ):
            log_event(conn, "rollback")
            state = S.IDLE

        # ── 3️⃣  Update background slowly to adapt to light ─────
        cv2.accumulateWeighted(roi.astype("float32"), bg, 0.001)

if __name__ == "__main__":
    main()
