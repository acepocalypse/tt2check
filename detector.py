#!/usr/bin/env python3
"""
detector.py  –  headless Streamlink version
────────────────────────────────────────────
• Streamlink serves a smooth local proxy of the Cedar Point HLS feed
• OpenCV reads the proxy (no freeze / fast-forward)
• FSM detects SUCCESS / ROLLBACK / INCOMPLETE
• Events are inserted into events.db (UTC)
"""

# ───────── stdlib
import subprocess, socket, signal, time, enum, sqlite3, pathlib, sys
from collections import deque

# ───────── third-party
import cv2
import numpy as np
import streamlink

# ──────── 0)  CONFIG  ──────────────────────────────────────────────
STREAM_PORT   = 8888                                   # local proxy port
HOST_URL      = "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"

ROI_BOT = (608, 761, 55, 97)   # bottom   (x,y,w,h)
ROI_MID = (674, 234,  8,189)   # mid tower
ROI_TOP = (505, 429, 22,109)   # descent side of crest

MOTION_ENTER = 900
MOTION_EXIT  = 800
TOP_THRESH   = 800
UP, DOWN     = -0.4, 0.4
WAIT_TIMEOUT = 30              # s

FRAME_W, FRAME_H = 1280, 720   # native
FPS_OUT = 10                   # OpenCV fps

DB_FILE = pathlib.Path("events.db")

# ──────── 1)  SQLite helper─────────────────────────────────────────
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS launches (
                       id INTEGER PRIMARY KEY,
                       ts_utc TEXT,
                       outcome TEXT)""")
    return conn

def log_event(conn, outcome: str):
    conn.execute("INSERT INTO launches VALUES (NULL, datetime('now'), ?)",
                 (outcome,))
    conn.commit()
    print(f"\n[event] {outcome.upper()}")

# ──────── 2)  Streamlink proxy launcher─────────────────────────────
def launch_streamlink(url: str) -> subprocess.Popen:
    cmd = [
        "streamlink", url, "best",
        "--player", sys.executable,             # dummy player
        "--player-external-http",
        "--player-external-http-port", str(STREAM_PORT),
        "--player-external-http-continuous",
        "--ringbuffer-size", "16M",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
    )

    # wait until proxy socket opens
    for _ in range(20):  # ≈10 s
        if proc.poll() is not None:
            raise RuntimeError("Streamlink exited early")
        with socket.socket() as s:
            if not s.connect_ex(("127.0.0.1", STREAM_PORT)):
                print("[stream] proxy up on :%d" % STREAM_PORT)
                return proc
        time.sleep(0.5)

    proc.kill()
    raise RuntimeError("proxy never came up")

STREAM_URL = f"http://127.0.0.1:{STREAM_PORT}/stream"

# ──────── 3)  Small helpers─────────────────────────────────────────
def centroid(mask) -> int | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None
    m = cv2.moments(c)
    return int(m["m01"] / m["m00"]) if m["m00"] else None

class S(enum.Enum):
    IDLE=0; ASC1=1; RBACK=2; WAIT=3; ASC3=4
COLOR = {S.IDLE:"\033[37m",S.ASC1:"\033[32m",S.RBACK:"\033[33m",
         S.WAIT:"\033[36m",S.ASC3:"\033[35m","END":"\033[0m"}

# ──────── 4)  Main routine──────────────────────────────────────────
def main():
    # start proxy
    sl_proc = launch_streamlink(HOST_URL)
    cap     = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sl_proc.terminate(); sl_proc.wait()
        sys.exit("OpenCV cannot read stream")

    conn   = db()
    state  = S.IDLE
    bg_bot = bg_top = None
    hist   = deque(maxlen=3)
    t0 = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # restart proxy once, then loop
                print("\n[warn] proxy hiccup – restarting …")
                sl_proc.kill(); sl_proc.wait()
                sl_proc = launch_streamlink(HOST_URL)
                cap.open(STREAM_URL)
                bg_bot = bg_top = None
                state = S.IDLE
                hist.clear()
                continue

            # ----- crop ROIs -----
            xb,yb,wb,hb = ROI_BOT
            bot = frame[yb:yb+hb, xb:xb+wb]
            xt,yt,wt,ht = ROI_TOP
            top = frame[yt:yt+ht, xt:xt+wt]

            if bg_bot is None:
                bg_bot, bg_top = bot.astype("float32"), top.astype("float32")
                continue

            # ----- bottom ROI -----
            diff_bot   = cv2.absdiff(bot, bg_bot.astype("uint8"))
            motion_bot = (diff_bot > 25).sum()
            cv2.accumulateWeighted(bot.astype("float32"), bg_bot, 0.01)

            gray  = cv2.cvtColor(diff_bot, cv2.COLOR_BGR2GRAY)
            _, msk = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            cy_loc = centroid(msk)
            cy_abs = yb + cy_loc if cy_loc is not None else None

            hist.append(cy_loc if cy_loc is not None else hist[-1] if hist else None)
            v = (hist[-1]-hist[0])/2 if len(hist)>=3 and None not in hist else 0

            # ----- crest ROI -----
            diff_top   = cv2.absdiff(top, bg_top.astype("uint8"))
            motion_top = (diff_top > 25).sum()
            cv2.accumulateWeighted(top.astype("float32"), bg_top, 0.01)

            # ----- live ticker -----
            print(f"\r{COLOR[state]}{state.name:<5}{COLOR['END']} "
                  f"bot={motion_bot:<4} top={motion_top:<4} v={v:+3.1f}", end="")

            # ----- FSM -----
            if state==S.IDLE and motion_bot>MOTION_ENTER and v<UP:
                state=S.ASC1

            elif state==S.ASC1 and v>DOWN and cy_loc is not None:
                state=S.RBACK

            elif state==S.RBACK and (motion_bot<MOTION_EXIT or cy_loc is None):
                state,t0=S.WAIT,time.time()

            elif state==S.WAIT:
                if motion_bot>MOTION_EXIT: t0=time.time()
                elif motion_bot>MOTION_ENTER and v<UP and cy_loc is not None:
                    state=S.ASC3; t_asc3=time.time()
                elif time.time()-t0>WAIT_TIMEOUT:
                    log_event(conn,"incomplete"); state=S.IDLE

            elif state==S.ASC3:
                if motion_top>TOP_THRESH and v>DOWN:
                    log_event(conn,"success");  state=S.IDLE
                elif v>DOWN and cy_abs is not None and cy_abs>ROI_MID[1]:
                    log_event(conn,"rollback"); state=S.IDLE
                elif time.time()-t_asc3>SUCCESS_WINDOW:
                    log_event(conn,"rollback"); state=S.IDLE

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sl_proc.terminate(); sl_proc.wait(timeout=5)
        conn.close()
        print("\n[bye] detector stopped")

# ──────── 5)  Entry point ─────
if __name__ == "__main__":
    main()
