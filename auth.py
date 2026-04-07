"""
Teacher auth: register/login with email+password → get API key.
Max 10 active teacher accounts.
Uses SQLite (no external DB needed).
"""
import os
import uuid
import sqlite3
import hashlib
import secrets
from functools import wraps
from flask import request, jsonify

DB_PATH = os.environ.get("DB_PATH", "/tmp/users.db")
MAX_USERS = 10


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teachers (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                email    TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                api_key  TEXT UNIQUE NOT NULL
            )
        """)
        conn.commit()


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# --- Auth decorator for protected routes ---
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "").strip()
        if not key:
            return jsonify({"error": "Missing X-API-Key header"}), 401
        with get_db() as conn:
            row = conn.execute(
                "SELECT id FROM teachers WHERE api_key = ?", (key,)
            ).fetchone()
        if not row:
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated


# --- Route handlers (registered in app.py) ---
def register():
    # Optional: lock registration behind an admin secret
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if admin_secret:
        provided = request.headers.get("X-Admin-Secret", "").strip()
        if provided != admin_secret:
            return jsonify({"error": "Registration is restricted. Contact admin."}), 403

    data = request.get_json(force=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
        if count >= MAX_USERS:
            return jsonify({"error": f"Max {MAX_USERS} teacher accounts reached"}), 403

        existing = conn.execute(
            "SELECT id FROM teachers WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            return jsonify({"error": "Email already registered"}), 409

        api_key = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO teachers (email, password, api_key) VALUES (?, ?, ?)",
            (email, _hash(password), api_key),
        )
        conn.commit()

    return jsonify({"message": "Registered successfully", "api_key": api_key}), 201


def login():
    data = request.get_json(force=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT api_key FROM teachers WHERE email = ? AND password = ?",
            (email, _hash(password)),
        ).fetchone()

    if not row:
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({"api_key": row["api_key"]}), 200


def regenerate_key():
    """Lets a teacher get a fresh API key using their email+password."""
    data = request.get_json(force=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM teachers WHERE email = ? AND password = ?",
            (email, _hash(password)),
        ).fetchone()
        if not row:
            return jsonify({"error": "Invalid email or password"}), 401

        new_key = secrets.token_urlsafe(32)
        conn.execute(
            "UPDATE teachers SET api_key = ? WHERE id = ?",
            (new_key, row["id"]),
        )
        conn.commit()

    return jsonify({"api_key": new_key}), 200
