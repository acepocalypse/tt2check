"""
detector.py
────────────────────────────────────────
Live Top Thrill 2 watcher – no Streamlink required
• connects to the Pixelcaster HLS feed (tries several hosts)
• runs the same FSM you just tuned
• logs SUCCESS / ROLLBACK / INCOMPLETE to SQLite (events.db)
"""

import cv2, time, enum, sqlite3, pathlib
from collections import deque

# ───────── stream hosts ─────────
HOSTS = [
    "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8",
    "https://cs3.pixelcaster.com/live/cedar2.stream/playlist.m3u8",
    "https://cs2.pixelcaster.com/live/cedar2.stream/playlist.m3u8",
    "http://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8",
    "http://cs3.pixelcaster.com/live/cedar2.stream/playlist.m3u8",
    "http://cs2.pixelcaster.com/live/cedar2.stream/playlist.m3u8",
]

def open_stream():
    for url in HOSTS:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f"[stream] connected → {url}")
            return cap
        cap.release()
    raise RuntimeError("all host attempts failed")

# ───────── ROIs (x, y, w, h) ─────────
ROI_BOT = (608, 761, 55, 97)   # bottom band
ROI_MID = (674, 234,  8,189)   # mid-tower band
ROI_TOP = (505, 429, 22,109)   # descent side of crest

# ───────── thresholds ─────────
MOTION_ENTER = 350
MOTION_EXIT  = 350
TOP_ENTER    = 300
UP,  DOWN    = -0.5, 0.5        # px/frame
WAIT_TIMEOUT = 30

# ───────── SQLite setup ───────
DB_PATH = pathlib.Path("events.db")
conn = sqlite3.connect(DB_PATH)
conn.execute("""CREATE TABLE IF NOT EXISTS launches (
                  id INTEGER PRIMARY KEY,
                  ts_utc TEXT,
                  outcome TEXT )""")
def log_event(outcome:str):
    conn.execute("INSERT INTO launches(ts_utc,outcome) VALUES (datetime('now'),?)",
                 (outcome,))
    conn.commit()
    print(f"\n[event] {outcome.upper()} logged")

# ───────── FSM & helpers ───────
class S(enum.Enum):
    IDLE=0; ASC1=1; RBACK=2; WAIT=3; ASC3=4

C = {S.IDLE:"\033[37m",S.ASC1:"\033[32m",S.RBACK:"\033[33m",
     S.WAIT:"\033[36m",S.ASC3:"\033[35m","END":"\033[0m"}

def centroid(mask):
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c=max(cnts,key=cv2.contourArea)
    if cv2.contourArea(c)<40: return None
    m=cv2.moments(c)
    return int(m["m01"]/m["m00"]) if m["m00"] else None

# ───────── main loop ───────────
cap   = open_stream()
state = S.IDLE
hist  = deque(maxlen=3)
bg_bot = bg_top = None
t0 = None

while True:
    ok, frame = cap.read()
    if not ok:
        print("\n[stream] lost – reconnecting…")
        time.sleep(1)
        cap.release()
        cap   = open_stream()
        bg_bot = bg_top = None
        state = S.IDLE
        hist.clear()
        continue

    # ---------- ROI crops ----------
    xb,yb,wb,hb = ROI_BOT
    bot = frame[yb:yb+hb, xb:xb+wb]
    xt,yt,wt,ht = ROI_TOP
    top = frame[yt:yt+ht, xt:xt+wt]

    # ---------- BG init ----------
    if bg_bot is None:
        bg_bot = bot.astype("float32")
        bg_top = top.astype("float32")
        continue

    # ---------- bottom ROI ----------
    diff_bot   = cv2.absdiff(bot, bg_bot.astype("uint8"))
    motion_bot = (diff_bot > 25).sum()
    cv2.accumulateWeighted(bot.astype("float32"), bg_bot, 0.001)

    gray_bot = cv2.cvtColor(diff_bot, cv2.COLOR_BGR2GRAY)
    _, msk   = cv2.threshold(gray_bot, 40, 255, cv2.THRESH_BINARY)
    cy_loc   = centroid(msk)                   # 0..hb-1 or None
    cy_abs   = yb + cy_loc if cy_loc is not None else None

    hist.append(cy_loc if cy_loc is not None else hist[-1] if hist else None)
    v = (hist[-1]-hist[0])/2 if len(hist)>=3 and None not in hist else 0

    # ---------- crest ROI motion ----------
    diff_top   = cv2.absdiff(top, bg_top.astype("uint8"))
    motion_top = (diff_top > 25).sum()
    cv2.accumulateWeighted(top.astype("float32"), bg_top, 0.001)

    # ---------- live console ----------
    print(f"\r{C[state]}{state.name:<5}{C['END']} "
          f"bot={motion_bot:<4} top={motion_top:<4} v={v:+4.1f}", end="")

    # ---------- FSM ----------
    if state==S.IDLE and motion_bot>MOTION_ENTER and v<UP:
        state=S.ASC1

    elif state==S.ASC1 and v>DOWN and cy_loc is not None:
        state=S.RBACK

    elif state==S.RBACK and (motion_bot<MOTION_EXIT or cy_loc is None):
        state,t0=S.WAIT,time.time()

    elif state==S.WAIT:
        if motion_bot>MOTION_EXIT:                     # still falling
            t0=time.time()
        elif motion_bot>MOTION_ENTER and v<UP and cy_loc is not None:
            state=S.ASC3
        elif time.time()-t0>WAIT_TIMEOUT:
            log_event("incomplete"); state=S.IDLE

    elif state==S.ASC3:
        if motion_top>TOP_ENTER and v>DOWN:
            log_event("success");  state=S.IDLE
        elif v>DOWN and cy_abs is not None \
             and cy_abs > ROI_MID[1]:
            log_event("rollback"); state=S.IDLE
