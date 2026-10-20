"""Cookie sessions for deployment; local mode retains the existing private ledger."""
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

COOKIE = "chagok_session"
MAX_AGE = 30 * 24 * 3600


def password_hash(password: str, salt=None):
    salt = salt or secrets.token_hex(16)
    value = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=32768, r=8, p=1, maxmem=64*1024*1024, dklen=32)
    return salt + ":" + value.hex()


class Credentials(BaseModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_]{3,32}$")
    password: str = Field(min_length=10, max_length=128)


class Auth:
    def __init__(self, store, production):
        self.store, self.production = store, production
        self.attempts, self.lock = {}, threading.Lock()

    def current(self, request: Request):
        token = request.cookies.get(COOKIE)
        if token:
            with self.store.connect() as db:
                row = db.execute("SELECT u.id,u.username FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?", (hashlib.sha256(token.encode()).hexdigest(), int(time.time()))).fetchone()
            if row:
                return dict(row)
        if not self.production:
            return {"id": "local", "username": "내 가계부"}
        return None

    def owner(self, request: Request):
        user = self.current(request)
        if user is None:
            raise HTTPException(401, "로그인하고 나만의 가계부를 시작하세요.")
        return user["id"]

    def limit(self, request, key=None, maximum=8):
        key = key or (request.client.host if request.client else "unknown")
        now = time.monotonic()
        with self.lock:
            self.attempts = {k:v for k,v in self.attempts.items() if v and v[-1] > now-60}
            times = [t for t in self.attempts.get(key, []) if t > now-60]
            if len(times) >= maximum:
                raise HTTPException(429, "잠시 후 다시 시도해주세요.")
            self.attempts[key] = times + [now]

    def issue(self, response, user_id):
        token = secrets.token_urlsafe(48)
        with self.store.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (int(time.time()),))
            db.execute("INSERT INTO sessions VALUES(?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), user_id, int(time.time())+MAX_AGE))
        response.set_cookie(COOKIE, token, httponly=True, secure=self.production, samesite="strict", max_age=MAX_AGE, path="/")

    def router(self):
        router = APIRouter(prefix="/api/auth")

        @router.get("/me")
        def me(request: Request):
            user = self.current(request)
            return {"authenticated": user is not None, "username": user["username"] if user else None, "mode": "public" if self.production else "local"}

        @router.post("/register", status_code=201)
        def register(body: Credentials, request: Request, response: Response):
            self.limit(request)
            if os.getenv("ALLOW_REGISTRATION", "true").lower() != "true":
                raise HTTPException(403, "현재 가입을 받지 않고 있습니다.")
            username, user_id = body.username.lower(), uuid.uuid4().hex
            encoded = password_hash(body.password)
            try:
                with self.store.connect() as db:
                    db.execute("INSERT INTO users VALUES(?,?,?,?)", (user_id, username, encoded, int(time.time())))
            except sqlite3.IntegrityError:
                raise HTTPException(409, "사용할 수 없는 아이디입니다.")
            self.issue(response, user_id)
            return {"username": username}

        @router.post("/login")
        def login(body: Credentials, request: Request, response: Response):
            self.limit(request)
            with self.store.connect() as db:
                row = db.execute("SELECT * FROM users WHERE username=?", (body.username.lower(),)).fetchone()
            # Also perform scrypt for an absent user to reduce enumeration timing.
            expected = row["password_hash"] if row else ("0"*32 + ":" + "0"*64)
            actual = password_hash(body.password, expected.split(":", 1)[0])
            if not row or not hmac.compare_digest(actual, expected):
                raise HTTPException(401, "아이디 또는 비밀번호를 확인해주세요.")
            self.issue(response, row["id"])
            return {"username": row["username"]}

        @router.post("/logout", status_code=204)
        def logout(request: Request, response: Response):
            token = request.cookies.get(COOKIE, "")
            with self.store.connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
            response.delete_cookie(COOKIE, path="/", secure=self.production, httponly=True, samesite="strict")

        return router
