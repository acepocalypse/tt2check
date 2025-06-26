#!/usr/bin/env python3
"""
app.py  –  FastAPI wrapper for Top Thrill 2 launch history
──────────────────────────────────────────────────────────
• Reads the same SQLite file (`events.db`) the detector writes.
• Endpoints
    GET  /latest            → most-recent event
    GET  /events?limit=100  → last N events (default 100, max 1000)
    GET  /stats             → count of each outcome
    GET  /healthz           → simple “OK” for load balancers / uptime checks
• CORS wide-open so a front-end or Twitter/X bot can call it.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3, pathlib, contextlib

# ───────── configuration ─────────
DB_PATH = pathlib.Path(__file__).with_name("events.db") 

app = FastAPI(
    title="Top Thrill 2 Launch API",
    version="1.0.0",
    description="Serves launch / rollback history collected by detector.py"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://topthrillcheck.netlify.app"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ───────── helpers ─────────
def db_conn():
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="events.db not found")
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def row_to_dict(row, cursor):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

# ───────── endpoints ─────────
@app.get("/")
def landing():
    return {
        "message": "Welcome to the Top Thrill 2 Launch API!",
        "endpoints": {
            "/latest": "Get the most recent event",
            "/events": "Get the last N events (default 100, max 1000)",
            "/stats": "Get the count of each outcome",
            "/healthz": "Check API health status"
        }
    }

@app.get("/latest")
def latest():
    with contextlib.closing(db_conn()) as conn:
        cur = conn.execute("SELECT * FROM launches ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            raise HTTPException(404, "no events yet")
        return row_to_dict(row, cur)

@app.get("/events")
def events(limit: int = Query(100, ge=1, le=1000)):
    with contextlib.closing(db_conn()) as conn:
        cur = conn.execute(
            "SELECT * FROM launches ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [row_to_dict(r, cur) for r in rows]

@app.get("/stats")
def stats():
    with contextlib.closing(db_conn()) as conn:
        cur = conn.execute("""
            SELECT outcome, COUNT(*) AS n
            FROM launches
            GROUP BY outcome
        """)
        return {row[0]: row[1] for row in cur.fetchall()}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
