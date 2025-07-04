#!/usr/bin/env python3
import cv2, enum, math, sqlite3, streamlink, time, argparse, pathlib, requests, json
from collections import deque

# ─────────────────── ROI COORDS ───────────────────
ROI_BOT    = (605, 775, 55, 97)
ROI_TOP    = (512, 409, 26, 70)
ROI_VERIFY = (581, 887, 30, 18)

xb, yb, wb, hb = ROI_BOT
xt, yt, wt, ht = ROI_TOP
xv, yv, wv, hv = ROI_VERIFY

ROI = {
    "bot_L":    (xb,           yb, wb//2-1, hb),
    "bot_R":    (xb+wb//2+1,   yb, wb//2-1, hb),
    "top_high": (xt,           yt,          wt, ht//2-1),
    "top_low":  (xt,           yt+ht//2+1,  wt, ht//2-1),
    "verify_L": (xv,           yv, wv//2-1, hv),
    "verify_R": (xv+wv//2+1,   yv, wv//2-1, hv),
}

# ─────────────────── CONSTANTS ───────────────────
BASE_FRAMES = 60
SIGMA_BOT, SIGMA_TOP, SIGMA_VERIFY = 10, 8, 6
ARM_DELAY = 3
VERIFY_WINDOW = 8.0
AUTO_SUCCESS = 10
ASC2_GRACE_PERIOD = 6.0
ROLLBACK_CONFIRM_FRAMES = 25
ROLLBACK_VELOCITY_THRESHOLD = 6.0
VERIFY_ROLLBACK_VELOCITY = 7.0
UP_FAST, DOWN_FAST = -1.0, 1.0
VELOCITY_SMOOTHING = 12
RECONNECT_GRACE_PERIOD = 60.0
LIVE_URL = "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"

QUEUE_TIMES_URL   = "https://queue-times.com/parks/50/queue_times.json"
TT2_RIDE_ID       = 3772
QUEUE_UPDATE_INTERVAL = 300
DB_PATH = pathlib.Path("events.db")
MIN_EVENT_INTERVAL = 81.0

# ─────────────────── UTILS ───────────────────
def open_source(path):
    if path and path.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise RuntimeError(f"cannot open {path}")
        return cap, False, cap.get(cv2.CAP_PROP_FPS) or 30
    try:
        url = streamlink.streams(LIVE_URL)["best"].url
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened(): raise RuntimeError("cannot open stream")
        return cap, True, 30
    except Exception as e:
        print(f"Stream error: {e}")
        raise RuntimeError(f"Stream connection failed: {e}")

def db():
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS launches(id INTEGER PRIMARY KEY, ts REAL, outcome TEXT)")
    c.execute("""CREATE TABLE IF NOT EXISTS queue_times(
        id INTEGER PRIMARY KEY, ts REAL, is_open BOOLEAN, wait_time INTEGER, last_updated TEXT
    )""")
    return c

def log_event(c, result, t):
    cursor = c.cursor()
    cursor.execute("SELECT MAX(ts) FROM launches WHERE outcome = ?", (result,))
    row = cursor.fetchone()
    last_ts = row[0] if row and row[0] else None
    if last_ts and time.time() - last_ts < MIN_EVENT_INTERVAL:
        return False
    cursor.execute("INSERT INTO launches VALUES(NULL, ?, ?)", (time.time(), result))
    c.commit(); print(f"\n[{result.upper():8} @ {t:7.2f}s]")
    return True

def fetch_queue_times():
    try:
        d = requests.get(QUEUE_TIMES_URL, timeout=10).json()
        for land in d.get('lands', []):
            for ride in land.get('rides', []):
                if ride.get("id") == TT2_RIDE_ID:
                    return {"is_open": ride.get("is_open", False),
                            "wait_time": ride.get("wait_time", 0),
                            "last_updated": ride.get("last_updated", "")}
    except Exception as e:
        print(f"queue-times error: {e}")
    return None

def log_queue_time(c, q, t):
    cursor = c.cursor()
    cursor.execute("INSERT INTO queue_times VALUES(NULL, ?, ?, ?, ?)",
              (t, q["is_open"], q["wait_time"], q["last_updated"]))
    c.commit()

def centroid(mask):
    cnt,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt: return None
    cnt = max(cnt, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 40: return None
    m = cv2.moments(cnt); return int(m["m01"]/m["m00"]) if m["m00"] else None

def smooth_velocity(hist):
    if len(hist) < VELOCITY_SMOOTHING: return 0
    pts = [(y,t) for y,t in list(hist)[-VELOCITY_SMOOTHING:] if y is not None]
    if len(pts) < 2: return 0
    dt = pts[-1][1] - pts[0][1]
    return (pts[-1][0]-pts[0][0])/dt if dt else 0

# ─────────────────── FSM ───────────────────
class S(enum.Enum):
    IDLE=0; ASC1=1; RBACK_DECEL=2; WAIT=3; ASC2=4; VERIFY=5

def detector(src, gui=True):
    conn = db()
    cap, live, fps = open_source(src)
    bg = {k:None for k in ROI}; base={k:[] for k in ROI}; thr={k:math.inf for k in ROI}
    armed=False; virtual=0.0; state=S.IDLE
    hist=deque(maxlen=VELOCITY_SMOOTHING)
    asc2_start=descent_start=verify_dead=None
    descent_seen=pending_id=False
    verify_hits=0
    rollback_confirm_count=0
    last_queue_update=0; queue_data=None
    reconnect_attempts=0
    max_reconnect_attempts=10
    start_time = time.time()
    last_reconnect_time = None
    last_event_time = 0
    
    frame_count = 0
    stream_fps = 10.0 if live else fps

    while True:
        now = time.time()
        ok, frame = cap.read()
        if not ok:
            if live: 
                reconnect_attempts += 1
                backoff_time = min(30, 2 ** min(reconnect_attempts, 5))
                print(f"\nStream disconnected, reconnecting in {backoff_time}s... (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                if reconnect_attempts > max_reconnect_attempts:
                    print("Max reconnection attempts reached, exiting")
                    break
                time.sleep(backoff_time)
                try:
                    cap.release()
                    cap, live, _ = open_source(src)
                    reconnect_attempts = 0
                    last_reconnect_time = frame_count / stream_fps
                    bg = {k:None for k in ROI}
                    base = {k:[] for k in ROI}
                    thr = {k:math.inf for k in ROI}
                    armed = False
                    state = S.IDLE
                    hist.clear()
                    asc2_start = descent_start = verify_dead = None
                    descent_seen = False
                    verify_hits = 0
                    rollback_confirm_count = 0
                    print("Reconnected successfully, resetting detector state")
                except Exception as e:
                    print(f"Reconnection failed: {e}")
                continue
            break

        frame_count += 1
        frame_time = frame_count / stream_fps
        
        if live and reconnect_attempts > 0:
            reconnect_attempts = 0

        relative_time = frame_time

        # queue API
        if live and now-last_queue_update>=QUEUE_UPDATE_INTERVAL:
            q=fetch_queue_times()
            if q: queue_data=q; log_queue_time(conn,q,now)
            last_queue_update=now

        # motion
        mot={}
        for k,(x,y,w,h) in ROI.items():
            sub=frame[y:y+h,x:x+w]
            if bg[k] is None: bg[k]=sub.astype("float32"); continue
            diff=cv2.absdiff(sub,bg[k].astype("uint8"))
            mot[k]=(diff>25).sum()
            cv2.accumulateWeighted(sub.astype("float32"),bg[k],0.02)
            if len(base[k])<BASE_FRAMES: base[k].append(mot[k])

        # ─── THRESHOLDS ───
        if not armed and all(len(v)>=BASE_FRAMES for v in base.values()) and relative_time>=ARM_DELAY:
            for k in ROI:
                m=sum(base[k])/len(base[k])
                s=(sum((v-m)**2 for v in base[k])/len(base[k]))**0.5
                if "top" in k: thr[k]=m+SIGMA_TOP*s
                elif "verify" in k: thr[k]=min(m+SIGMA_VERIFY*s, wv*hv*0.6)
                else: thr[k]=m+SIGMA_BOT*s*1.2
            armed=True
            print(f"\n[ARMED at t={relative_time:.2f}s]")

        bot_hot = armed and mot["bot_L"]>thr["bot_L"] and mot["bot_R"]>thr["bot_R"]
        top_hot = armed and (mot["top_high"]>thr["top_high"] or mot["top_low"]>thr["top_low"])
        ver_hot = armed and mot["verify_L"]>thr["verify_L"] and mot["verify_R"]>thr["verify_R"]

        # descent tracking
        if top_hot:
            descent_start=frame_time; descent_seen=True
        elif descent_start and frame_time-descent_start>1.0:
            descent_start=None; descent_seen=False

        # velocity
        bx,by,bw,bh=ROI["bot_L"]
        diff=cv2.absdiff(frame[by:by+bh,bx:bx+bw],bg["bot_L"].astype("uint8"))
        _,mk=cv2.threshold(cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY),40,255,cv2.THRESH_BINARY)
        cy=centroid(mk)
        hist.append((cy,frame_time) if cy is not None else (hist[-1][0],frame_time) if hist else (None,frame_time))
        v=smooth_velocity(hist)

        in_grace = asc2_start is not None and frame_time-asc2_start<ASC2_GRACE_PERIOD
        can_rb = not in_grace or ver_hot
        reconnect_grace = last_reconnect_time is not None and frame_time - last_reconnect_time < RECONNECT_GRACE_PERIOD
        stuck_in_rback = state is S.RBACK_DECEL and frame_time - getattr(detector, '_rback_start', frame_time) > 30

        # ─── FSM ───
        if   state is S.IDLE and bot_hot and v<UP_FAST: 
            state=S.ASC1; rollback_confirm_count=0
        elif state is S.ASC1 and bot_hot and v>DOWN_FAST: 
            state=S.RBACK_DECEL; rollback_confirm_count=0; detector._rback_start = frame_time
        elif state is S.RBACK_DECEL and (mot["bot_L"]<thr["bot_L"]*0.1 and mot["bot_R"]<thr["bot_R"]*0.1 or stuck_in_rback):
            t_wait=frame_time; state=S.WAIT; rollback_confirm_count=0
        elif state is S.WAIT and bot_hot and v<UP_FAST and frame_time-t_wait>0.5:
            state=S.ASC2; asc2_start=frame_time; descent_seen=False; rollback_confirm_count=0
        elif state is S.ASC2 and descent_seen:
            pending_id=True; verify_dead=frame_time+VERIFY_WINDOW; state=S.VERIFY; verify_hits=0; rollback_confirm_count=0
        elif state is S.ASC2 and can_rb and bot_hot and v>DOWN_FAST*3.0 and not descent_seen and not reconnect_grace:
            if v > 5.0:
                rollback_confirm_count += 1
            else:
                rollback_confirm_count = max(0, rollback_confirm_count - 2)
            if rollback_confirm_count >= 15 and frame_time - last_event_time >= MIN_EVENT_INTERVAL:
                if log_event(conn,"rollback",frame_time): 
                    last_event_time = frame_time
                state=S.IDLE; rollback_confirm_count=0
        elif state is S.ASC2 and frame_time-asc2_start>=AUTO_SUCCESS:
            if frame_time - last_event_time >= MIN_EVENT_INTERVAL:
                if log_event(conn,"success",frame_time):
                    last_event_time = frame_time
            state=S.IDLE; rollback_confirm_count=0
        elif state is S.VERIFY:
            rollback_detected = False
            if bot_hot and v>ROLLBACK_VELOCITY_THRESHOLD and v < 12.0:
                verify_hits+=1; rollback_detected=True
            elif ver_hot and v>VERIFY_ROLLBACK_VELOCITY and v < 15.0:
                verify_hits+=2; rollback_detected=True

            if not rollback_detected:
                verify_hits=max(0,verify_hits-4)
                
            if verify_hits>=ROLLBACK_CONFIRM_FRAMES and not reconnect_grace and frame_time - last_event_time >= MIN_EVENT_INTERVAL:
                if log_event(conn,"rollback",frame_time):
                    last_event_time = frame_time
                state=S.IDLE; rollback_confirm_count=0
            elif frame_time>=verify_dead:
                if frame_time - last_event_time >= MIN_EVENT_INTERVAL:
                    if log_event(conn,"success",frame_time):
                        last_event_time = frame_time
                state=S.IDLE; rollback_confirm_count=0
        
        # Reset rollback counter if not in rollback-detecting states or if velocity is too extreme
        if state not in [S.ASC2, S.VERIFY] or abs(v) > 20.0:
            rollback_confirm_count = 0
            
        # More conservative reconnect grace handling - favor success
        if reconnect_grace:
            if top_hot or (state in [S.ASC2, S.VERIFY] and frame_time - last_reconnect_time > 5):
                if frame_time - last_event_time >= MIN_EVENT_INTERVAL:
                    if log_event(conn,"success",frame_time):
                        last_event_time = frame_time
                state=S.IDLE; rollback_confirm_count=0

        if state is S.IDLE:
            asc2_start=descent_start=verify_dead=None
            descent_seen=False; verify_hits=0; rollback_confirm_count=0

        # ─── GUI ───
        if gui:
            view=frame.copy()
            for tag,(x,y,w,h) in {"BOT":ROI_BOT,"RBK":ROI_VERIFY,"TOP":ROI_TOP}.items():
                color=(0,255,0) if tag!="TOP" else (0,0,255)
                thick=2 if (tag=="BOT" and bot_hot) or (tag=="RBK" and ver_hot) or (tag=="TOP" and top_hot) else 1
                cv2.rectangle(view,(x,y),(x+w,y+h),color,thick)
                cv2.putText(view,tag,(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
            lines=[f"State:{state.name}",
                   f"v:{v:.2f}  grace:{'Y' if in_grace else 'N'}",
                   f"RC:{rollback_confirm_count}"]
            if queue_data:
                lines.append(f"TT2:{'OPEN' if queue_data['is_open'] else 'CLOSED'} "
                             f"{queue_data['wait_time']}m")
            for i,t in enumerate(lines):
                cv2.putText(view,t,(10,30+i*18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            cv2.imshow("TT2 detector",view)
            if cv2.waitKey(1)&0xFF==27: break

        print(f"\r{state.name:<8} B={'Y'if bot_hot else'-'} T={'Y'if top_hot else'-'} V={'Y'if ver_hot else'-'} "
              f"v={v:5.2f} {'GR' if in_grace else '  '} RC:{rollback_confirm_count:2d} t={relative_time:7.2f} "
              f"{'ARMED' if armed else f'base:{min(len(v) for v in base.values())}/{BASE_FRAMES}'}",end='')

    cap.release(); conn.close()
    if gui: cv2.destroyAllWindows(); print("\n[bye]")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="TT2 detector")
    ap.add_argument("--video"); ap.add_argument("--no-gui",action="store_true")
    args = ap.parse_args()
    detector(args.video, gui=not args.no_gui)
