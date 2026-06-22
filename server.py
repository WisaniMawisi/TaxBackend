""""TaxApp FastAPI backend.

Modules:
- /api/auth          - JWT email/password authentication (httpOnly cookies)
- /api/income        - monthly income + PAYE records
- /api/expenses      - manual or OCR-sourced expense items
- /api/slips         - receipt image upload + OCR extraction (GPT-4o vision)
- /api/tax/summary   - annual SARS calculation (taxable income / refund / owed)
"""
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import os
import uuid
import logging
import mimetypes
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal

import bcrypt
import jwt
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response, UploadFile, File, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field, ConfigDict

from tax_engine import annual_summary, calculate_annual_tax, SARS_BRACKETS_2024_2025
try:
    from ocr_service import extract_receipt, CATEGORIES
except Exception:
    extract_receipt = None
    CATEGORIES = ["Transport", "Medical", "Education", "Business", "Other"]
    
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("taxapp")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALG = "HS256"
ACCESS_TTL_MIN = 60 * 12         # 12h - cookie-based, comfortable for a finance dashboard
REFRESH_TTL_DAYS = 14
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "/app/storage/slips"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@taxapp.za")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Admin@2026")

CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if not CORS_ORIGINS:
    CORS_ORIGINS = ["*"]

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="TaxApp API")
api = APIRouter(prefix="/api")

# CORS: when credentials are sent, origin must be explicit (not "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded slip images
app.mount("/storage/slips", StaticFiles(directory=str(STORAGE_DIR)), name="slips")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_token(payload: dict, expires_minutes: int) -> str:
    data = payload.copy()
    data["exp"] = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALG)


def set_auth_cookies(response: Response, user_id: str, email: str) -> None:
    access = create_token({"sub": user_id, "email": email, "type": "access"}, ACCESS_TTL_MIN)
    refresh = create_token({"sub": user_id, "type": "refresh"}, REFRESH_TTL_DAYS * 24 * 60)
    common = dict(httponly=True, secure=True, samesite="none", path="/")
    response.set_cookie("access_token", access, max_age=ACCESS_TTL_MIN * 60, **common)
    response.set_cookie("refresh_token", refresh, max_age=REFRESH_TTL_DAYS * 86400, **common)


async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"id": payload["sub"]})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        user.pop("_id", None)
        user.pop("password_hash", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def strip_id(doc: dict) -> dict:
    if doc is None:
        return doc
    doc.pop("_id", None)
    return doc


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = Field(min_length=1, max_length=80)
    age: int = Field(default=30, ge=18, le=120)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: EmailStr
    name: str
    age: int = 30
    role: str = "user"
    created_at: Optional[datetime] = None


class IncomeIn(BaseModel):
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2000, le=2100)
    income: float = Field(ge=0)
    tax_paid: float = Field(ge=0)
    note: Optional[str] = Field(default=None, max_length=200)


class IncomeOut(IncomeIn):
    id: str
    user_id: str
    created_at: datetime


class ExpenseIn(BaseModel):
    amount: float = Field(gt=0)
    date: str  # ISO YYYY-MM-DD
    vendor: str = Field(min_length=1, max_length=120)
    category: Literal["Transport", "Medical", "Education", "Business", "Other"] = "Other"
    notes: Optional[str] = Field(default=None, max_length=300)


class ExpenseOut(ExpenseIn):
    id: str
    user_id: str
    source: str  # "Manual" | "OCR"
    slip_id: Optional[str] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# Startup: seed admin + indexes
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    await db.users.create_index("email", unique=True)
    await db.users.create_index("id", unique=True)
    await db.login_attempts.create_index("identifier")
    await db.income_records.create_index([("user_id", 1), ("year", 1), ("month", 1)], unique=True)
    await db.expenses.create_index([("user_id", 1), ("date", -1)])
    await db.slips.create_index([("user_id", 1), ("uploaded_at", -1)])

    # Seed admin
    existing = await db.users.find_one({"email": ADMIN_EMAIL})
    if existing is None:
        await db.users.insert_one({
            "id": str(uuid.uuid4()),
            "email": ADMIN_EMAIL,
            "password_hash": hash_password(ADMIN_PASSWORD),
            "name": "Admin",
            "age": 35,
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Seeded admin user %s", ADMIN_EMAIL)
    elif not verify_password(ADMIN_PASSWORD, existing.get("password_hash", "")):
        await db.users.update_one(
            {"email": ADMIN_EMAIL},
            {"$set": {"password_hash": hash_password(ADMIN_PASSWORD)}},
        )
        logger.info("Rotated admin password")


@app.on_event("shutdown")
async def shutdown() -> None:
    client.close()


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
auth_router = APIRouter(prefix="/auth", tags=["auth"])


async def _check_lockout(identifier: str) -> None:
    rec = await db.login_attempts.find_one({"identifier": identifier})
    if not rec:
        return
    if rec.get("count", 0) >= 5:
        last = rec.get("last_attempt")
        if isinstance(last, str):
            last = datetime.fromisoformat(last)
        if last and datetime.now(timezone.utc) - last < timedelta(minutes=15):
            raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 15 minutes.")
        # cooldown elapsed
        await db.login_attempts.delete_one({"identifier": identifier})


async def _record_failed(identifier: str) -> None:
    await db.login_attempts.update_one(
        {"identifier": identifier},
        {"$inc": {"count": 1}, "$set": {"last_attempt": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )


@auth_router.post("/register", response_model=UserOut)
async def register(payload: RegisterIn, response: Response):
    email = payload.email.lower()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    doc = {
        "id": user_id,
        "email": email,
        "name": payload.name,
        "age": payload.age,
        "role": "user",
        "password_hash": hash_password(payload.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(doc)
    set_auth_cookies(response, user_id, email)
    return UserOut(id=user_id, email=email, name=payload.name, age=payload.age,
                   role="user", created_at=datetime.fromisoformat(doc["created_at"]))


@auth_router.post("/login", response_model=UserOut)
async def login(payload: LoginIn, request: Request, response: Response):
    email = payload.email.lower()
    ip = request.client.host if request.client else "unknown"
    identifier = f"{ip}:{email}"
    await _check_lockout(identifier)

    user = await db.users.find_one({"email": email})
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        await _record_failed(identifier)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    await db.login_attempts.delete_one({"identifier": identifier})
    set_auth_cookies(response, user["id"], email)
    return UserOut(
        id=user["id"], email=user["email"], name=user.get("name", ""),
        age=user.get("age", 30), role=user.get("role", "user"),
        created_at=datetime.fromisoformat(user["created_at"]) if isinstance(user.get("created_at"), str) else user.get("created_at"),
    )


@auth_router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return {"ok": True}


@auth_router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)):
    created = user.get("created_at")
    if isinstance(created, str):
        created = datetime.fromisoformat(created)
    return UserOut(
        id=user["id"], email=user["email"], name=user.get("name", ""),
        age=user.get("age", 30), role=user.get("role", "user"),
        created_at=created,
    )


@auth_router.post("/refresh")
async def refresh(request: Request, response: Response):
    rt = request.cookies.get("refresh_token")
    if not rt:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        payload = jwt.decode(rt, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        user = await db.users.find_one({"id": payload["sub"]})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        set_auth_cookies(response, user["id"], user["email"])
        return {"ok": True}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


api.include_router(auth_router)


# ---------------------------------------------------------------------------
# Income endpoints
# ---------------------------------------------------------------------------
income_router = APIRouter(prefix="/income", tags=["income"])


@income_router.post("", response_model=IncomeOut)
async def create_income(payload: IncomeIn, user: dict = Depends(get_current_user)):
    doc = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "month": payload.month,
        "year": payload.year,
        "income": float(payload.income),
        "tax_paid": float(payload.tax_paid),
        "note": payload.note,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # Upsert by (user_id, year, month)
    await db.income_records.update_one(
        {"user_id": user["id"], "year": payload.year, "month": payload.month},
        {"$set": doc},
        upsert=True,
    )
    saved = await db.income_records.find_one({"user_id": user["id"], "year": payload.year, "month": payload.month})
    saved = strip_id(saved)
    saved["created_at"] = datetime.fromisoformat(saved["created_at"]) if isinstance(saved["created_at"], str) else saved["created_at"]
    return IncomeOut(**saved)


@income_router.get("", response_model=List[IncomeOut])
async def list_income(year: Optional[int] = None, user: dict = Depends(get_current_user)):
    q = {"user_id": user["id"]}
    if year is not None:
        q["year"] = year
    cursor = db.income_records.find(q).sort([("year", 1), ("month", 1)])
    items: List[IncomeOut] = []
    async for d in cursor:
        d = strip_id(d)
        d["created_at"] = datetime.fromisoformat(d["created_at"]) if isinstance(d["created_at"], str) else d["created_at"]
        items.append(IncomeOut(**d))
    return items


@income_router.delete("/{income_id}")
async def delete_income(income_id: str, user: dict = Depends(get_current_user)):
    res = await db.income_records.delete_one({"id": income_id, "user_id": user["id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Income record not found")
    return {"ok": True}


api.include_router(income_router)


# ---------------------------------------------------------------------------
# Expense endpoints
# ---------------------------------------------------------------------------
expense_router = APIRouter(prefix="/expenses", tags=["expenses"])


@expense_router.post("", response_model=ExpenseOut)
async def create_expense(payload: ExpenseIn, user: dict = Depends(get_current_user)):
    doc = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "amount": float(payload.amount),
        "date": payload.date,
        "vendor": payload.vendor,
        "category": payload.category,
        "notes": payload.notes,
        "source": "Manual",
        "slip_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.expenses.insert_one(doc)
    doc = strip_id(doc)
    doc["created_at"] = datetime.fromisoformat(doc["created_at"])
    return ExpenseOut(**doc)


@expense_router.get("", response_model=List[ExpenseOut])
async def list_expenses(
    year: Optional[int] = None,
    month: Optional[int] = None,
    category: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    q: dict = {"user_id": user["id"]}
    if category:
        q["category"] = category
    cursor = db.expenses.find(q).sort([("date", -1), ("created_at", -1)])
    items: List[ExpenseOut] = []
    async for d in cursor:
        d = strip_id(d)
        if year is not None and not d.get("date", "").startswith(str(year)):
            continue
        if month is not None:
            m = f"{month:02d}"
            if not d.get("date", "")[5:7] == m:
                continue
        d["created_at"] = datetime.fromisoformat(d["created_at"]) if isinstance(d["created_at"], str) else d["created_at"]
        items.append(ExpenseOut(**d))
    return items


@expense_router.patch("/{expense_id}", response_model=ExpenseOut)
async def update_expense(expense_id: str, payload: ExpenseIn, user: dict = Depends(get_current_user)):
    update = {
        "amount": float(payload.amount),
        "date": payload.date,
        "vendor": payload.vendor,
        "category": payload.category,
        "notes": payload.notes,
    }
    res = await db.expenses.update_one({"id": expense_id, "user_id": user["id"]}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Expense not found")
    doc = await db.expenses.find_one({"id": expense_id, "user_id": user["id"]})
    doc = strip_id(doc)
    doc["created_at"] = datetime.fromisoformat(doc["created_at"]) if isinstance(doc["created_at"], str) else doc["created_at"]
    return ExpenseOut(**doc)


@expense_router.delete("/{expense_id}")
async def delete_expense(expense_id: str, user: dict = Depends(get_current_user)):
    res = await db.expenses.delete_one({"id": expense_id, "user_id": user["id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Expense not found")
    return {"ok": True}


api.include_router(expense_router)


# ---------------------------------------------------------------------------
# Slip / OCR endpoints
# ---------------------------------------------------------------------------
slip_router = APIRouter(prefix="/slips", tags=["slips"])

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


@slip_router.post("/upload")
async def upload_slip(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Upload a receipt image, run OCR, auto-create the linked expense."""
    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or ""
    if mime not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime}. Use JPEG/PNG/WEBP.")

    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}[mime]
    slip_id = str(uuid.uuid4())
    file_name = f"{slip_id}.{ext}"
    file_path = STORAGE_DIR / file_name

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")
    with open(file_path, "wb") as fh:
        fh.write(contents)

    slip_doc = {
        "id": slip_id,
        "user_id": user["id"],
        "file_path": str(file_path),
        "file_url": f"/storage/slips/{file_name}",
        "mime": mime,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "processed": False,
        "ocr_result": None,
        "expense_id": None,
    }
    await db.slips.insert_one(slip_doc)

    # Run OCR (best-effort)
    ocr_data = None
    ocr_error = None
    try:
        ocr_data = await extract_receipt(str(file_path), mime)
    except Exception as exc:
        logger.warning("OCR failed for slip %s: %s", slip_id, exc)
        ocr_error = str(exc)

    expense_id = None
    if ocr_data:
        expense_id = str(uuid.uuid4())
        await db.expenses.insert_one({
            "id": expense_id,
            "user_id": user["id"],
            "amount": float(ocr_data.get("amount") or 0),
            "date": ocr_data.get("date"),
            "vendor": ocr_data.get("vendor") or "Unknown",
            "category": ocr_data.get("category") or "Other",
            "notes": ocr_data.get("notes") or "",
            "source": "OCR",
            "slip_id": slip_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    await db.slips.update_one(
        {"id": slip_id},
        {"$set": {
            "processed": ocr_data is not None,
            "ocr_result": ocr_data,
            "ocr_error": ocr_error,
            "expense_id": expense_id,
        }},
    )

    slip_doc.update({
        "processed": ocr_data is not None,
        "ocr_result": ocr_data,
        "ocr_error": ocr_error,
        "expense_id": expense_id,
    })
    slip_doc.pop("file_path", None)
    return slip_doc


@slip_router.get("")
async def list_slips(user: dict = Depends(get_current_user)):
    cursor = db.slips.find({"user_id": user["id"]}).sort([("uploaded_at", -1)])
    items = []
    async for d in cursor:
        d = strip_id(d)
        d.pop("file_path", None)
        items.append(d)
    return items


@slip_router.delete("/{slip_id}")
async def delete_slip(slip_id: str, user: dict = Depends(get_current_user)):
    slip = await db.slips.find_one({"id": slip_id, "user_id": user["id"]})
    if not slip:
        raise HTTPException(status_code=404, detail="Slip not found")
    # remove file on disk
    try:
        p = Path(slip.get("file_path", ""))
        if p.exists():
            p.unlink()
    except Exception:
        pass
    await db.slips.delete_one({"id": slip_id, "user_id": user["id"]})
    # Also remove the linked expense
    if slip.get("expense_id"):
        await db.expenses.delete_one({"id": slip["expense_id"], "user_id": user["id"]})
    return {"ok": True}


api.include_router(slip_router)


# ---------------------------------------------------------------------------
# Tax engine endpoints
# ---------------------------------------------------------------------------
tax_router = APIRouter(prefix="/tax", tags=["tax"])


@tax_router.get("/brackets")
async def get_brackets():
    return {"year": "2024/2025", "brackets": SARS_BRACKETS_2024_2025}


@tax_router.get("/summary")
async def tax_summary(year: int = 2025, user: dict = Depends(get_current_user)):
    incomes = []
    async for d in db.income_records.find({"user_id": user["id"], "year": year}):
        d = strip_id(d)
        incomes.append(d)
    expenses = []
    async for d in db.expenses.find({"user_id": user["id"]}):
        d = strip_id(d)
        if d.get("date", "").startswith(str(year)):
            expenses.append(d)
    summary = annual_summary(incomes, expenses, age=user.get("age", 30))
    return {"year": year, **summary, "monthly_records": incomes}


api.include_router(tax_router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@api.get("/")
async def root():
    return {"app": "TaxApp API", "status": "ok"}


app.include_router(api)
