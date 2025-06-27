#!/usr/bin/env python3
"""
Top Thrill 2 launch / rollback detector (live m3u8 or MP4).

Logic
=====
ASC1  → RBACK_DECEL → WAIT → ASC2
        └── if crest held ≥ 0.4 s → VERIFY (5 s)
              ├── bottom ROI hot + v DOWN → ROLLBACK
              ├── window timeout         → SUCCESS
ASC2 with no verdict for 30 s → SUCCESS

TOP ROI is split (top_high / top_low); either slice lighting counts.
Direction of motion is **ignored** for crest detection.
"""

import cv2, enum, math, sqlite3, streamlink, time, argparse, pathlib
from collections import deque

# ───────────────────────────────────────── CONSTANTS
ROI_BOT = (598, 775, 55, 97)
ROI_TOP = (505, 409, 22, 70)
xb, yb, wb, hb = ROI_BOT
xt, yt, wt, ht = ROI_TOP
ROI = {
    "bot_L":   (xb,             yb, wb//2-1, hb),
    "bot_R":   (xb+wb//2+1,     yb, wb//2-1, hb),
    "top_high":(xt,             yt,          wt, ht//2-1),
    "top_low": (xt,             yt+ht//2+1,  wt, ht//2-1),
}

BASE_FRAMES       = 60
SIGMA_BOT, SIGMA_TOP = 6, 4
ARM_DELAY         = 3
CREST_HOLD        = 0.40     # s
VERIFY_WINDOW     = 5.0      # s
AUTO_SUCCESS      = 30.0     # s after ASC2
UP_FAST, DOWN_FAST= -0.6, 0.6
LIVE_URL          = "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"
DB_PATH           = pathlib.Path("events.db")

# ───────────────────────────────────────── UTILITIES
def open_source(path):
    if path and path.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError(f"cannot open {path}")
        return cap, False, cap.get(cv2.CAP_PROP_FPS) or 30
    url = streamlink.streams(LIVE_URL)["best"].url
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened(): raise RuntimeError("cannot open stream")
    return cap, True, 30

def db():
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS launches(id INTEGER PRIMARY KEY, ts REAL, outcome TEXT)")
    return c

def log_event(c, outcome, t):
    cur = c.execute("INSERT INTO launches VALUES(NULL, ?, ?)", (t, outcome))
    c.commit()
    print(f"\n[{outcome.upper():8} @ {t:7.2f}s]")
    return cur.lastrowid

def centroid(mask):
    cnt,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt: return None
    cnt = max(cnt, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 40: return None
    m = cv2.moments(cnt)
    return int(m["m01"]/m["m00"]) if m["m00"] else None

class S(enum.Enum):
    IDLE=0; ASC1=1; RBACK_DECEL=2; WAIT=3; ASC2=4; VERIFY=5

# ───────────────────────────────────────── DETECTOR
def detector(src, gui=True):
    conn = db()
    cap, live, fps = open_source(src)
    bg   = {k: None for k in ROI}
    base = {k: []  for k in ROI}
    thr  = {k: math.inf for k in ROI}

    armed = False
    virtual = 0.0
    state = S.IDLE
    hist  = deque(maxlen=3)

    crest_start = verify_dead = asc2_start = None
    pending_id  = None

    while True:
        ok, frame = cap.read()
        if not ok:
            if live:
                time.sleep(3); cap, live, _ = open_source(src)
                continue
            break

        now = time.time() if live else (virtual := virtual + 1 / fps)

        # ── motion & background
        mot = {}
        for k, (x, y, w, h) in ROI.items():
            sub = frame[y:y+h, x:x+w]
            if bg[k] is None:
                bg[k] = sub.astype("float32")
                continue
            diff = cv2.absdiff(sub, bg[k].astype("uint8"))
            mot[k] = (diff > 25).sum()
            cv2.accumulateWeighted(sub.astype("float32"), bg[k], 0.02)
            if len(base[k]) < BASE_FRAMES:
                base[k].append(mot[k])

        if not armed and all(len(v) >= BASE_FRAMES for v in base.values()) and now >= ARM_DELAY:
            for k in ROI:
                m = sum(base[k]) / len(base[k])
                s = (sum((v - m) ** 2 for v in base[k]) / len(base[k])) ** 0.5
                thr[k] = m + (SIGMA_TOP if "top" in k else SIGMA_BOT) * s
            armed = True

        bot_hot = armed and mot.get("bot_L", 0) > thr["bot_L"] and mot.get("bot_R", 0) > thr["bot_R"]
        top_hot = armed and sum(mot.get(k, 0) > thr[k] for k in ("top_high", "top_low")) >= 1

        # velocity on bottom ROI
        bx, by, bw, bh = ROI["bot_L"]
        diff = cv2.absdiff(frame[by:by+bh, bx:bx+bw], bg["bot_L"].astype("uint8"))
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mk = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        cy = centroid(mk)
        hist.append((cy, now) if cy is not None else (hist[-1][0], now) if hist else (None, now))
        if len(hist) >= 3 and None not in [h[0] for h in hist]:
            dt = hist[-1][1] - hist[0][1]
            v = (hist[-1][0] - hist[0][0]) / dt if dt else 0
        else:
            v = 0

        # crest hold (direction ignored)
        if top_hot:
            crest_start = crest_start or now
        else:
            crest_start = None
        crest_ok = crest_start and (now - crest_start) >= CREST_HOLD

        # ── FSM
        if state is S.IDLE and bot_hot and v < UP_FAST:
            state = S.ASC1
        elif state is S.ASC1 and bot_hot and v > DOWN_FAST:
            state = S.RBACK_DECEL
        elif state is S.RBACK_DECEL and all(mot.get(k, 0) < thr[k] * 0.1 for k in ("bot_L", "bot_R")):
            t_wait = now; state = S.WAIT
        elif state is S.WAIT and bot_hot and v < UP_FAST and now - t_wait > 0.5:
            state = S.ASC2; asc2_start = now
        elif state is S.ASC2 and crest_ok:
            pending_id  = log_event(conn, "pending", now)
            verify_dead = now + VERIFY_WINDOW
            state = S.VERIFY
        elif state is S.ASC2 and bot_hot and v > DOWN_FAST:
            log_event(conn, "rollback", now)
            state = S.IDLE; asc2_start = None
        elif state is S.ASC2 and asc2_start and now - asc2_start >= AUTO_SUCCESS:
            log_event(conn, "success", now)
            state = S.IDLE; asc2_start = None
        elif state is S.VERIFY:
            if bot_hot and v > DOWN_FAST:
                conn.execute("UPDATE launches SET outcome='rollback' WHERE id=?", (pending_id,))
                conn.commit()
                state = S.IDLE
            elif now >= verify_dead:
                conn.execute("UPDATE launches SET outcome='success'  WHERE id=?", (pending_id,))
                conn.commit()
                state = S.IDLE

        # ── GUI
        if gui:
            view = frame.copy()
            for k, (x, y, w, h) in ROI.items():
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0) if "bot" in k else (0, 0, 255), 1)
            cv2.putText(view, state.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("TT2 detector", view)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        print(f"\r{state.name:<12} BOT={'Y' if bot_hot else '-'} TOP={'Y' if top_hot else '-'}  t={now:7.2f}s", end='')

    cap.release(); conn.close()
    if gui: cv2.destroyAllWindows()
    print("\n[bye]")

# ───────────────────────────────────────── CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TT2 detector (live or MP4)")
    ap.add_argument("--video", help="MP4 file; omit for live stream")
    ap.add_argument("--no-gui", action="store_true", help="disable video window")
    detector(ap.parse_args().video, gui=not ap.parse_args().no_gui)
