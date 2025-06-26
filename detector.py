#!/usr/bin/env python3
import enum, time, sqlite3, pathlib, cv2, streamlink
from collections import deque

HOST_URL = "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"

ROI_BOT  = (608, 761, 55, 97)
ROI_TOP  = (505, 429, 22,115)

#!/usr/bin/env python3
import enum, time, sqlite3, pathlib, cv2, streamlink
from collections import deque

HOST_URL = "https://cs4.pixelcaster.com/live/cedar2.stream/playlist.m3u8"

ROI_BOT  = (608, 761, 55, 97)
ROI_TOP  = (505, 429, 22,125)

ENTER_THR, EXIT_THR, TOP_THR = 750, 650, 900
UP, DOWN = -0.4, 0.4
WAIT_TIMEOUT, MIN_WAIT_TIME, MIN_ASC1_TIME, COOLDOWN = 45, 10, 1, 60
ARM_DELAY = 3

DB = pathlib.Path("events.db")

def db():
    conn = sqlite3.connect(DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS launches("
        "id INTEGER PRIMARY KEY, ts REAL, outcome TEXT)"
    )
    return conn

def log_event(conn, outcome):
    ts = time.time()
    conn.execute("INSERT INTO launches VALUES(NULL, ?, ?)", (ts, outcome))
    conn.commit()
    print(f"\n[event] {outcome.upper()} @ {time.strftime('%H:%M:%S', time.localtime(ts))}")

def correct_rollback(conn):
    conn.execute(
        "UPDATE launches SET outcome='success' "
        "WHERE id = (SELECT id FROM launches "
        "WHERE outcome='rollback' ORDER BY id DESC LIMIT 1)"
    )
    conn.commit()
    print("\n[correction] rollback → success")

def open_stream(url):
    sl_url = streamlink.streams(url)["best"].url
    cap = cv2.VideoCapture(sl_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV failed to open HLS URL")
    print("[stream] connected")
    return cap

def centroid(mask):
    c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not c:
        return None
    c = max(c, key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None
    m = cv2.moments(c)
    return int(m["m01"] / m["m00"]) if m["m00"] else None

class S(enum.Enum):
    IDLE = 0
    ASC1 = 1
    RBACK = 2
    WAIT = 3
    ASC3 = 4

def main(headless=True):
    conn = db()
    cap = open_stream(HOST_URL)

    state = S.IDLE
    bg_bot = bg_top = None
    hist = deque(maxlen=3)

    t0 = t_asc1 = None
    logged = False
    last_rb = 0
    last_bg = 0
    last_success = 0
    quiet_t = None
    top_hi = 0
    ready_at = time.time() + ARM_DELAY

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                time.sleep(3)
                cap = open_stream(HOST_URL)
                bg_bot = bg_top = None
                state = S.IDLE
                hist.clear()
                ready_at = time.time() + ARM_DELAY
                continue

            now = time.time()

            xb, yb, wb, hb = ROI_BOT
            bot = frame[yb: yb + hb, xb: xb + wb]
            xt, yt, wt, ht = ROI_TOP
            top = frame[yt: yt + ht, xt: xt + wt]

            if bg_bot is None:
                bg_bot, bg_top = bot.astype("float32"), top.astype("float32")
                last_bg = now
                continue

            diff_bot = cv2.absdiff(bot, bg_bot.astype("uint8"))
            diff_top = cv2.absdiff(top, bg_top.astype("uint8"))
            motion_bot = (diff_bot > 25).sum()
            motion_top = (diff_top > 25).sum()

            if now - last_bg > 0.5:
                cv2.accumulateWeighted(bot.astype("float32"), bg_bot, 0.02)
                cv2.accumulateWeighted(top.astype("float32"), bg_top, 0.02)
                last_bg = now

            gray = cv2.cvtColor(diff_bot, cv2.COLOR_BGR2GRAY)
            _, msk = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            cy = centroid(msk)

            if cy is not None:
                hist.append((cy, now))
            elif hist:
                hist.append((hist[-1][0], now))
            else:
                hist.append((None, now))

            if len(hist) >= 3 and None not in [h[0] for h in hist]:
                dt = hist[-1][1] - hist[0][1]
                v = (hist[-1][0] - hist[0][0]) / dt if dt else 0
            else:
                v = 0

            # crest / success detector with 2-frame confirmation
            if state != S.IDLE:
                top_hi = top_hi + 1 if motion_top > TOP_THR else 0
                if top_hi >= 2 and not logged and now >= ready_at:
                    if now - last_success >= COOLDOWN:
                        if last_rb and now - last_rb < 30:
                            correct_rollback(conn)
                        log_event(conn, "success")
                        last_success = now
                    state = S.IDLE
                    logged = True
                    quiet_t = None
                    top_hi = 0

            # FSM transitions
            if state == S.IDLE:
                if motion_bot > ENTER_THR and v < UP:
                    state, t_asc1 = S.ASC1, now
            elif state == S.ASC1 and v > DOWN and now - t_asc1 > MIN_ASC1_TIME:
                state = S.RBACK
            elif state == S.RBACK and motion_bot < EXIT_THR:
                state, t0 = S.WAIT, now
            elif state == S.WAIT:
                if motion_bot > ENTER_THR and v < UP and now - t0 > MIN_WAIT_TIME:
                    state = S.ASC3
                elif now - t0 > WAIT_TIMEOUT and now >= ready_at:
                    log_event(conn, "incomplete")
                    state = S.IDLE
                    logged = True
            elif state == S.ASC3 and v > DOWN and now >= ready_at:
                log_event(conn, "rollback")
                state = S.IDLE
                logged = True
                last_rb = now

            # quiet-time latch
            if motion_bot < 50 and motion_top < 50:
                quiet_t = quiet_t or now
            else:
                quiet_t = None
            if quiet_t and now - quiet_t > 5:
                logged = False

            # live ticker
            print(f"\r{state.name:<5} bot={motion_bot:<5} top={motion_top:<5} v={v:+4.1f}", end="", flush=True)

            if not headless and cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        conn.close()
        if not headless:
            cv2.destroyAllWindows()
        print("\n[bye] detector stopped")

if __name__ == "__main__":
    main(headless=True)
