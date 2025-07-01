#!/usr/bin/env python3
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3, pathlib, contextlib

DB_PATH = pathlib.Path(__file__).with_name("events.db")

app = FastAPI(
    title="Top Thrill 2 Launch API",
    version="2.0.0",
    description="Launch / rollback and queue-time history collected by detector.py"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

def conn():
    if not DB_PATH.exists():
        raise HTTPException(500, detail="events.db missing")
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

# ───────── launch endpoints ─────────
@app.get("/latest")
def latest():
    with contextlib.closing(conn()) as c:
        row = c.execute("SELECT * FROM launches ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            raise HTTPException(404, "no events yet")
        return dict(row)

@app.get("/events")
def events(limit: int = Query(100, ge=1, le=1000)):
    with contextlib.closing(conn()) as c:
        rows = c.execute("SELECT * FROM launches ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

@app.get("/stats")
def stats():
    with contextlib.closing(conn()) as c:
        rows = c.execute("SELECT outcome, COUNT(*) n FROM launches GROUP BY outcome").fetchall()
        return {r["outcome"]: r["n"] for r in rows}

# ───────── queue-time endpoints ─────────
@app.get("/queue/latest")
def queue_latest():
    with contextlib.closing(conn()) as c:
        row = c.execute("SELECT * FROM queue_times ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            raise HTTPException(404, "no queue data yet")
        return dict(row)

@app.get("/queue")
def queue(limit: int = Query(100, ge=1, le=1000)):
    with contextlib.closing(conn()) as c:
        rows = c.execute("SELECT * FROM queue_times ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

# ───────── landing ─────────
@app.get("/")
def root():
    return {
        "message": "Welcome to the Top Thrill 2 Launch API",
        "version": "2.0.0",
        "endpoints": {
            "latest": "/latest - Get the most recent launch event",
            "events": "/events - Get launch events history",
            "stats": "/stats - Get launch outcome statistics",
            "queue_latest": "/queue/latest - Get the most recent queue time",
            "queue": "/queue - Get queue times history",
            "health": "/healthz - Health check"
        }
    }

# ───────── health ─────────
@app.get("/healthz")
def health():
    return {"status": "ok"}
