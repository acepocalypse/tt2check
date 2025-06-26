#!/usr/bin/env python3
"""
detector.py  –  your “good” Streamlink+OpenCV logic, head-less
"""

import enum, time, sqlite3, pathlib, sys
from collections import deque

import cv2
import numpy as np
import streamlink

# ───────── Config (unchanged ROIs / thresholds) ─────────
HOST_URL    = "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"
ROI_BOT     = (608, 761, 55, 97)
ROI_TOP     = (505, 429, 22,109)

ENTER_THR   = 750      # ASC1
EXIT_THR    = 650     # end rollback
TOP_THR     = 650     # crest hit
UP, DOWN    = -0.4, 0.4
WAIT_TIMEOUT = 45

DB = pathlib.Path("events.db")

# ───────── SQLite helper ─────────
def db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS launches(
                      id INTEGER PRIMARY KEY,
                      ts_utc TEXT,
                      outcome TEXT)""")
    return conn
def log_event(conn,outcome:str):
    conn.execute("INSERT INTO launches VALUES(NULL, datetime('now'), ?)",
                 (outcome,))
    conn.commit()
    print(f"\n[event] {outcome.upper()}")

# ───────── open via Streamlink ─────
def open_stream(url:str)->cv2.VideoCapture:
    print(f"[stream] resolving {url}")
    sl_url = streamlink.streams(url)["best"].url
    cap    = cv2.VideoCapture(sl_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open HLS URL")
    print("[stream] OpenCV connected")
    return cap

# ───────── centroid helper ─────────
def centroid(mask)->int|None:
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c=max(cnts,key=cv2.contourArea)
    if cv2.contourArea(c)<40: return None
    m=cv2.moments(c)
    return int(m["m01"]/m["m00"]) if m["m00"] else None

# ───────── FSM setup ─────────
class S(enum.Enum):
    IDLE=0; ASC1=1; RBACK=2; WAIT=3; ASC3=4
CLR = {S.IDLE:"\033[37m",S.ASC1:"\033[32m",S.RBACK:"\033[33m",
       S.WAIT:"\033[36m",S.ASC3:"\033[35m","END":"\033[0m"}

def main(headless=True):
    conn   = db()
    cap    = open_stream(HOST_URL)
    state  = S.IDLE
    bg_bot = bg_top = None
    hist   = deque(maxlen=3)
    t0=t_asc3=None
    logged_this_run = False  # Track if we've already logged for this run

    try:
        while True:
            ok, f = cap.read()
            if not ok:
                print("\n[warn] frame lost, reconnecting…")
                cap.release(); time.sleep(3)
                cap=open_stream(HOST_URL)
                bg_bot=bg_top=None; state=S.IDLE; hist.clear(); continue

            xb,yb,wb,hb = ROI_BOT
            bot = f[yb:yb+hb, xb:xb+wb]
            xt,yt,wt,ht = ROI_TOP
            top = f[yt:yt+ht, xt:xt+wt]

            if bg_bot is None:
                bg_bot, bg_top = bot.astype("float32"), top.astype("float32")
                continue

            # motion
            diff_bot = cv2.absdiff(bot,bg_bot.astype("uint8"))
            motion_bot = (diff_bot>25).sum()
            diff_top = cv2.absdiff(top,bg_top.astype("uint8"))
            motion_top = (diff_top>25).sum()

            if state in (S.IDLE,S.WAIT):
                cv2.accumulateWeighted(bot.astype("float32"),bg_bot,0.01)
                cv2.accumulateWeighted(top.astype("float32"),bg_top,0.01)

            gray=cv2.cvtColor(diff_bot,cv2.COLOR_BGR2GRAY)
            _,msk=cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
            cy_loc=centroid(msk)
            cy=yb+cy_loc if cy_loc is not None else None
            hist.append(cy_loc if cy_loc is not None else hist[-1] if hist else None)
            v=(hist[-1]-hist[0])/2 if len(hist)>=3 and None not in hist else 0

            # ticker
            print(f"\r{CLR[state]}{state.name:<5}{CLR['END']} "
                  f"bot={motion_bot:<4} top={motion_top:<4} v={v:+3.1f}",end="")

            # Check for TOP ROI hit in any state (except IDLE) - mark as success
            if state != S.IDLE and motion_top > TOP_THR and not logged_this_run:
                log_event(conn, "success")
                state = S.IDLE
                logged_this_run = False
                continue

            # FSM
            if state==S.IDLE:
                logged_this_run = False
                if motion_bot>ENTER_THR and v<UP:
                    state=S.ASC1
            elif state==S.ASC1 and v>DOWN and cy_loc is not None:
                state=S.RBACK
            elif state==S.RBACK and (motion_bot<EXIT_THR or cy_loc is None):
                state,t0=S.WAIT,time.time()
            elif state==S.WAIT:
                if motion_bot>ENTER_THR and v<UP and cy_loc is not None:
                    state=S.ASC3; t_asc3=time.time()
                elif time.time()-t0>WAIT_TIMEOUT:
                    log_event(conn,"incomplete"); state=S.IDLE
                    logged_this_run = True
            elif state==S.ASC3:
                if v>DOWN and cy is not None:
                    log_event(conn,"rollback"); state=S.IDLE
                    logged_this_run = True

            # optional preview
            if not headless:
                cv2.rectangle(f,(xb,yb),(xb+wb,yb+hb),(0,255,0),1)
                cv2.rectangle(f,(xt,yt),(xt+wt,yt+ht),(0,0,255),1)
                cv2.imshow("preview",f)
                if cv2.waitKey(1)&0xFF==27: break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); conn.close()
        if not headless: cv2.destroyAllWindows()
        print("\n[bye] detector stopped")

if __name__=="__main__":
    main(headless=True)
