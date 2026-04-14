from fastapi import FastAPI, APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import io
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Optional
import uuid
import secrets
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt as pyjwt
from emergentintegrations.llm.chat import LlmChat, UserMessage
from bs4 import BeautifulSoup
from fpdf import FPDF
import httpx
import dns.resolver
import smtplib
import re

# ============ App Setup ============

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Single app instance — no duplicates
app = FastAPI(title="FitnessAI API")
api_router = APIRouter(prefix="/api")

# ============ CORS — configured ONCE, before any routes ============
# List every origin that is allowed to call your API.
# Add your Vercel URL here. Do NOT use "*" with credentials=True.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    os.environ.get("FRONTEND_URL", "https://your-fitness-app.vercel.app"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Database — single global connection ============
mongo_url = os.environ.get("MONGO_URL")
db_name = os.environ.get("DB_NAME")
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

LLM_API_KEY = os.environ.get("LLM_API_KEY")
JWT_SECRET = os.environ.get("JWT_SECRET")
JWT_ALGORITHM = "HS256"

# ============ Password & JWT Helpers ============

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
        "type": "access",
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "type": "refresh",
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# ============ Email Verification ============

def validate_email_format(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def verify_email_smtp(email: str) -> dict:
    domain = email.split('@')[1]
    try:
        mx_records = dns.resolver.resolve(domain, 'MX')
        mx_host = str(mx_records[0].exchange).rstrip('.')
    except Exception:
        return {"valid": False, "reason": "Domain has no mail server."}
    try:
        smtp = smtplib.SMTP(timeout=10)
        smtp.connect(mx_host, 25)
        smtp.helo("fitnessai.com")
        smtp.mail("verify@fitnessai.com")
        code, _ = smtp.rcpt(email)
        smtp.quit()
        if code == 250:
            return {"valid": True, "reason": "Email accepted"}
        elif code == 550:
            return {"valid": False, "reason": "Email address does not exist"}
        return {"valid": True, "reason": "Server responded but could not confirm"}
    except Exception:
        return {"valid": True, "reason": "Domain verified, mailbox check unavailable"}

# ============ Models ============

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class ChatRequest(BaseModel):
    message: str
    chat_session_id: Optional[str] = None

class ChatResponse(BaseModel):
    message_id: str
    content: str
    intent: str
    chat_session_id: str

class ProfileUpdate(BaseModel):
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    gender: Optional[str] = None
    fitness_goal: Optional[str] = None
    activity_level: Optional[str] = None
    dietary_preference: Optional[str] = None

class ProgressEntry(BaseModel):
    weight: Optional[float] = None
    bmi: Optional[float] = None
    body_fat: Optional[float] = None
    muscle_mass: Optional[float] = None
    calories_burned: Optional[int] = None
    workout_minutes: Optional[int] = None

# ============ Knowledge Base ============

KNOWLEDGE_BASE = {
    "WORKOUT": """
## Trusted Workout Guidelines (WHO, ACSM, Department of Health)

**Aerobic Exercise:**
- Adults (18-64): 150-300 min moderate OR 75-150 min vigorous aerobic activity/week (WHO 2020)
- Include variety: walking, running, swimming, cycling
- Break up prolonged sitting every 30-60 minutes

**Strength Training:**
- 2+ days/week targeting all major muscle groups (ACSM)
- Progressive overload: gradually increase weight/reps
- Allow 48h recovery between same muscle groups
- Compound exercises (squats, deadlifts, bench press) are most efficient

**Flexibility & Balance:**
- Stretch major muscle groups 2-3 days/week
- Hold static stretches 15-60 seconds
- Include balance exercises, especially for older adults

**Recovery:**
- Rest days are essential for muscle repair and growth
- Sleep 7-9 hours for optimal recovery
- Active recovery (light walking, yoga) on rest days
- Hydrate adequately before, during, and after exercise

**Common Programs:**
- Push/Pull/Legs split for intermediate lifters
- Full body 3x/week for beginners
- HIIT 2-3x/week for fat loss
- 5x5 program for strength building
""",
    "NUTRITION": """
## Trusted Nutrition Guidelines (WHO, USDA, Department of Health)

**Macronutrients:**
- Protein: 0.8g/kg (sedentary) to 1.6-2.2g/kg (active/muscle building)
- Carbohydrates: 45-65% of total calories; focus on complex carbs
- Fats: 20-35% of total calories; prioritize unsaturated fats
- Fiber: 25-30g/day minimum

**Hydration:**
- Men: ~3.7L total water/day; Women: ~2.7L/day
- Additional 400-800ml per hour of exercise

**Meal Timing:**
- Protein every 3-4 hours for muscle synthesis
- Pre-workout: Carbs + protein 1-2 hours before
- Post-workout: Protein within 2 hours after exercise

**Food Sources:**
- Lean proteins: chicken, fish, eggs, legumes, tofu
- Complex carbs: oats, quinoa, sweet potato, brown rice
- Healthy fats: avocado, olive oil, nuts, fatty fish
- Vegetables: aim for 5+ servings of varied colors daily
""",
    "DIETARY_ADVICE": """
## Trusted Dietary Guidelines (WHO, CDC, Department of Health)

**Weight Loss:**
- Create 500-750 calorie deficit for 0.5-1kg loss/week (CDC)
- Never go below 1200 cal (women) or 1500 cal (men)
- High protein helps preserve muscle during deficit

**Weight/Muscle Gain:**
- Surplus of 250-500 calories above maintenance
- 1.6-2.2g protein per kg body weight
- Aim for 0.25-0.5kg gain per week

**Food Safety & Habits:**
- Limit added sugars to <10% of daily calories (WHO)
- Reduce sodium to <2000mg/day (WHO)
- Limit processed and ultra-processed foods

**Supplements:**
- Creatine: 3-5g/day, well-researched for strength
- Whey protein: convenient for meeting protein goals
- Multivitamin as insurance, not replacement for whole foods
""",
    "GENERAL_HEALTH": """
## Trusted General Health Guidelines (WHO, NIH, Department of Health)

**Sleep:**
- Adults need 7-9 hours quality sleep (NIH)
- Consistent sleep schedule improves quality
- Avoid screens 1 hour before bed

**Stress Management:**
- Chronic stress increases cortisol, promotes fat storage
- Regular exercise is one of the best stress relievers
- Meditation: even 10 min/day shows benefits

**Preventive Health:**
- Regular health check-ups annually
- Monitor blood pressure, cholesterol, blood sugar
- BMI is a rough guide; body composition matters more
"""
}

# ============ Web Scraping RAG Engine ============

SCRAPE_SOURCES = {
    "WORKOUT": [
        {"url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity", "name": "WHO Physical Activity"},
        {"url": "https://www.cdc.gov/physical-activity-basics/guidelines/adults.html", "name": "CDC Exercise Guidelines"},
    ],
    "NUTRITION": [
        {"url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet", "name": "WHO Healthy Diet"},
        {"url": "https://www.myplate.gov/eat-healthy/what-is-myplate", "name": "USDA MyPlate"},
    ],
    "DIETARY_ADVICE": [
        {"url": "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight", "name": "WHO Obesity Guidelines"},
        {"url": "https://www.cdc.gov/nutrition/php/data-research/index.html", "name": "CDC Dietary Guidelines"},
    ],
    "GENERAL_HEALTH": [
        {"url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity", "name": "WHO Health Guidelines"},
        {"url": "https://www.cdc.gov/sleep/about/index.html", "name": "CDC Sleep Health"},
    ],
}

async def scrape_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as c:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FitnessAI/1.0)"}
            response = await c.get(url, headers=headers)
            if response.status_code != 200:
                return ""
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
            return "\n".join(lines[:80])
    except Exception as e:
        logger.warning(f"Scrape failed for {url}: {e}")
        return ""

async def get_scraped_knowledge(intent: str) -> str:
    cache_key = f"scrape_cache_{intent}"
    cached = await db.scrape_cache.find_one({"cache_key": cache_key}, {"_id": 0})
    if cached:
        cached_at = cached.get("cached_at", "")
        if isinstance(cached_at, str):
            cached_at = datetime.fromisoformat(cached_at)
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        if (datetime.now(timezone.utc) - cached_at).total_seconds() < 86400:
            return cached.get("content", "")

    sources = SCRAPE_SOURCES.get(intent, SCRAPE_SOURCES["GENERAL_HEALTH"])
    scraped_parts = []
    for source in sources:
        content = await scrape_url(source["url"])
        if content:
            scraped_parts.append(f"### Source: {source['name']}\n{content[:1500]}")
    combined = "\n\n".join(scraped_parts)
    if combined:
        await db.scrape_cache.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "content": combined,
                "sources": [s["name"] for s in sources],
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )
    return combined

# ============ PDF Generator ============

class FitnessPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(255, 59, 48)
        self.cell(0, 12, "FitnessAI", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_draw_color(39, 39, 42)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(161, 161, 170)
        self.cell(0, 10, f"Generated by FitnessAI | Page {self.page_no()}", align="C")

def generate_meal_pdf(plan: dict, user_name: str = "User") -> bytes:
    pdf = FitnessPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(17, 17, 17)
    pdf.cell(0, 10, f"Meal Plan - {plan.get('date', 'Today')}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(161, 161, 170)
    pdf.cell(0, 6, f"Prepared for: {user_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    total_cal = total_protein = total_carbs = total_fats = 0
    for meal in plan.get("meals", []):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 149, 0)
        pdf.cell(0, 8, meal.get("meal_type", ""), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 7, meal.get("name", ""), new_x="LMARGIN", new_y="NEXT")
        if meal.get("description"):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, meal["description"])
        cal, protein, carbs, fats = meal.get("calories", 0), meal.get("protein", 0), meal.get("carbs", 0), meal.get("fats", 0)
        total_cal += cal; total_protein += protein; total_carbs += carbs; total_fats += fats
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 122, 255)
        pdf.cell(0, 6, f"{cal} cal  |  P: {protein}g  |  C: {carbs}g  |  F: {fats}g", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    pdf.ln(5)
    pdf.set_draw_color(39, 39, 42)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(255, 59, 48)
    pdf.cell(0, 8, "Daily Totals", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, f"Calories: {total_cal}  |  Protein: {total_protein}g  |  Carbs: {total_carbs}g  |  Fats: {total_fats}g", new_x="LMARGIN", new_y="NEXT")
    return pdf.output()

def generate_workout_pdf(workout: dict, user_name: str = "User") -> bytes:
    pdf = FitnessPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(17, 17, 17)
    status = "COMPLETED" if workout.get("completed") else "PENDING"
    pdf.cell(0, 10, f"Workout Plan - {workout.get('date', 'Today')}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(161, 161, 170)
    pdf.cell(0, 6, f"Prepared for: {user_name}  |  Status: {status}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(26, 26, 26)
    pdf.set_text_color(161, 161, 170)
    for h, w in [("Exercise", 60), ("Sets", 25), ("Reps", 30), ("Rest", 25), ("Muscle Group", 45)]:
        pdf.cell(w, 8, h, border=1, fill=True, align="C")
    pdf.ln()
    for i, ex in enumerate(workout.get("exercises", [])):
        bg = (240, 240, 240) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(60, 8, ex.get("name", "")[:30], border=1, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(25, 8, str(ex.get("sets", "")), border=1, fill=True, align="C")
        pdf.cell(30, 8, str(ex.get("reps", "")), border=1, fill=True, align="C")
        pdf.cell(25, 8, str(ex.get("rest", "")), border=1, fill=True, align="C")
        pdf.cell(45, 8, ex.get("muscle_group", "")[:20], border=1, fill=True, align="C")
        pdf.ln()
        if ex.get("tips"):
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(185, 6, f"  Tip: {ex['tips']}", new_x="LMARGIN", new_y="NEXT")
    return pdf.output()

# ============ Auth Helpers ============

async def get_current_user(request: Request) -> str:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload["sub"]
        user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user_id
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_user_profile_context(user_id: str) -> str:
    profile = await db.user_profiles.find_one({"user_id": user_id}, {"_id": 0})
    if not profile:
        return ""
    parts = []
    for key, label in [("age", "Age"), ("gender", "Gender"), ("height", "Height"),
                        ("weight", "Weight"), ("fitness_goal", "Fitness Goal"),
                        ("activity_level", "Activity Level"), ("dietary_preference", "Dietary Preference")]:
        if profile.get(key):
            suffix = "cm" if key == "height" else "kg" if key == "weight" else ""
            parts.append(f"{label}: {profile[key]}{suffix}")
    return "\n\nUser Profile:\n" + "\n".join(parts) if parts else ""

def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    response.set_cookie("access_token", access_token, httponly=True, secure=True, samesite="none", max_age=3600, path="/")
    response.set_cookie("refresh_token", refresh_token, httponly=True, secure=True, samesite="none", max_age=604800, path="/")

# ============ RAG & Intent ============

async def classify_intent(message: str) -> str:
    try:
        chat = LlmChat(
            api_key=LLM_API_KEY,
            session_id=f"intent_{uuid.uuid4().hex[:8]}",
            system_message="You are an intent classifier for a fitness app. Classify the user's message into exactly ONE category: WORKOUT, NUTRITION, DIETARY_ADVICE, GENERAL_HEALTH, or OTHER. Respond with ONLY the category name.",
        ).with_model("gemini", "gemini-2.0-flash")
        response = await chat.send_message(UserMessage(text=message))
        intent = response.strip().upper().replace(" ", "_")
        valid = ["WORKOUT", "NUTRITION", "DIETARY_ADVICE", "GENERAL_HEALTH", "OTHER"]
        return intent if intent in valid else "OTHER"
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        return "OTHER"

async def generate_rag_response(message: str, intent: str, user_id: str) -> str:
    try:
        if intent == "OTHER":
            system_msg = "You are FitnessAI, a fitness and health coach. The user asked something outside your scope. Respond in 1-2 sentences: politely say you only help with fitness, nutrition, and health topics, then suggest one relevant fitness question they could ask. Keep it under 40 words."
            chat = LlmChat(api_key=LLM_API_KEY, session_id=f"rag_{uuid.uuid4().hex[:8]}", system_message=system_msg).with_model("gemini", "gemini-2.0-flash")
            return await chat.send_message(UserMessage(text=message))

        static_knowledge = KNOWLEDGE_BASE.get(intent, KNOWLEDGE_BASE["GENERAL_HEALTH"])
        profile_context = await get_user_profile_context(user_id)
        scraped_knowledge = await get_scraped_knowledge(intent)
        scraped_section = f"\n\n## Live Data from Trusted Health Sources:\n{scraped_knowledge[:3000]}" if scraped_knowledge else ""

        system_msg = f"""You are FitnessAI, a knowledgeable and encouraging AI fitness coach. You provide evidence-based advice using trusted health information.

## Trusted Knowledge Base:
{static_knowledge}
{scraped_section}
{profile_context}

## Response Format Rules (MUST FOLLOW):
- Structure your response with clear sections using markdown headings (## and ###)
- Use bullet points (-) for lists
- Put key numbers, sets, reps, calories in **bold**
- Use tables when comparing options or showing structured data
- Keep each section concise — 3-5 bullet points max
- Add a brief motivational line at the end
- Cite sources when relevant (e.g., "According to WHO...")
- If giving a plan, organize as: Overview → Details → Tips → Disclaimer
- Use --- between major sections
- Never use emoji characters"""

        chat = LlmChat(api_key=LLM_API_KEY, session_id=f"rag_{uuid.uuid4().hex[:8]}", system_message=system_msg).with_model("gemini", "gemini-2.0-flash")
        return await chat.send_message(UserMessage(text=message))
    except Exception as e:
        logger.error(f"RAG response error: {e}")
        return "I'm having trouble processing your request. Please try again in a moment."

# ============ Startup / Shutdown ============

@app.on_event("startup")
async def startup_event():
    logger.info("Running startup tasks...")
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@fitnessai.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "FitnessAI@2026")
    existing = await db.users.find_one({"email": admin_email}, {"_id": 0})
    if not existing:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one({
            "user_id": user_id,
            "email": admin_email,
            "name": "Admin",
            "password_hash": hash_password(admin_password),
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Admin seeded: {admin_email}")
    elif not verify_password(admin_password, existing.get("password_hash", "")):
        await db.users.update_one({"email": admin_email}, {"$set": {"password_hash": hash_password(admin_password)}})
        logger.info("Admin password updated")
    await db.users.create_index("email", unique=True)
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Closing MongoDB connection...")
    client.close()

# ============ Health Check ============

@api_router.get("/")
async def root():
    return {"message": "FitnessAI API is running"}

# ============ Auth Endpoints ============

@api_router.post("/auth/register")
async def register(req: RegisterRequest, response: Response):
    email = req.email.strip().lower()
    if not email or not req.password or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Email and password (min 6 chars) required")
    if not validate_email_format(email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    domain = email.split("@")[1]
    if domain not in ["gmail.com", "googlemail.com"]:
        raise HTTPException(status_code=400, detail="Only Gmail addresses are allowed.")
    verification = verify_email_smtp(email)
    if not verification["valid"]:
        raise HTTPException(status_code=400, detail=f"Gmail not found: {verification['reason']}")
    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    user_doc = {
        "user_id": user_id, "email": email, "name": req.name.strip(),
        "password_hash": hash_password(req.password), "role": "user",
        "email_verified": True, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(user_doc)
    access_token = create_access_token(user_id, email)
    refresh_token = create_refresh_token(user_id)
    set_auth_cookies(response, access_token, refresh_token)
    return {"user_id": user_id, "email": email, "name": req.name.strip(), "role": "user", "created_at": user_doc["created_at"]}

@api_router.post("/auth/login")
async def login(req: LoginRequest, response: Response):
    email = req.email.strip().lower()
    user = await db.users.find_one({"email": email}, {"_id": 0})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    access_token = create_access_token(user["user_id"], email)
    refresh_token = create_refresh_token(user["user_id"])
    set_auth_cookies(response, access_token, refresh_token)
    return {"user_id": user["user_id"], "email": user["email"], "name": user["name"], "role": user.get("role", "user"), "created_at": user["created_at"]}

@api_router.get("/auth/me")
async def get_me(request: Request):
    user_id = await get_current_user(request)
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0, "password_hash": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    return user_doc

@api_router.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return {"message": "Logged out"}

@api_router.post("/auth/refresh")
async def refresh_token_endpoint(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload["sub"]
        user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        new_access = create_access_token(user_id, user["email"])
        response.set_cookie("access_token", new_access, httponly=True, secure=True, samesite="none", max_age=3600, path="/")
        return {"message": "Token refreshed"}
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

# ============ Password Reset ============

@api_router.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    email = req.email.strip().lower()
    user = await db.users.find_one({"email": email}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email")
    reset_token = secrets.token_urlsafe(32)
    await db.password_reset_tokens.insert_one({
        "token": reset_token, "user_id": user["user_id"], "email": email,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "used": False, "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return {"message": "Password reset link generated", "reset_token": reset_token, "expires_in": "1 hour"}

@api_router.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    if not req.new_password or len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    token_doc = await db.password_reset_tokens.find_one({"token": req.token, "used": False}, {"_id": 0})
    if not token_doc:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    expires_at = token_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Reset link has expired.")
    await db.users.update_one({"user_id": token_doc["user_id"]}, {"$set": {"password_hash": hash_password(req.new_password)}})
    await db.password_reset_tokens.update_one({"token": req.token}, {"$set": {"used": True}})
    return {"message": "Password reset successfully."}

# ============ Profile Endpoints ============

@api_router.get("/profile")
async def get_profile(request: Request):
    user_id = await get_current_user(request)
    profile = await db.user_profiles.find_one({"user_id": user_id}, {"_id": 0})
    if not profile:
        return {"user_id": user_id, "age": None, "height": None, "weight": None,
                "gender": None, "fitness_goal": None, "activity_level": None, "dietary_preference": None}
    return profile

@api_router.put("/profile")
async def update_profile(profile: ProfileUpdate, request: Request):
    user_id = await get_current_user(request)
    update_data = {k: v for k, v in profile.model_dump().items() if v is not None}
    update_data["user_id"] = user_id
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    await db.user_profiles.update_one({"user_id": user_id}, {"$set": update_data}, upsert=True)
    return await db.user_profiles.find_one({"user_id": user_id}, {"_id": 0})

# ============ Chat Endpoints ============

@api_router.post("/chat/message", response_model=ChatResponse)
async def send_chat_message(chat_request: ChatRequest, request: Request):
    user_id = await get_current_user(request)
    chat_session_id = chat_request.chat_session_id or f"session_{uuid.uuid4().hex[:12]}"
    user_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    await db.chat_messages.insert_one({
        "message_id": user_msg_id, "user_id": user_id, "chat_session_id": chat_session_id,
        "role": "user", "content": chat_request.message, "intent": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    intent = await classify_intent(chat_request.message)
    ai_response = await generate_rag_response(chat_request.message, intent, user_id)
    ai_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    await db.chat_messages.insert_one({
        "message_id": ai_msg_id, "user_id": user_id, "chat_session_id": chat_session_id,
        "role": "assistant", "content": ai_response, "intent": intent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return ChatResponse(message_id=ai_msg_id, content=ai_response, intent=intent, chat_session_id=chat_session_id)

@api_router.get("/chat/history/{chat_session_id}")
async def get_chat_history(chat_session_id: str, request: Request):
    user_id = await get_current_user(request)
    return await db.chat_messages.find(
        {"user_id": user_id, "chat_session_id": chat_session_id}, {"_id": 0}
    ).sort("timestamp", 1).to_list(1000)

@api_router.get("/chat/sessions")
async def get_chat_sessions(request: Request):
    user_id = await get_current_user(request)
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {
            "_id": "$chat_session_id",
            "last_message": {"$last": "$content"},
            "last_timestamp": {"$last": "$timestamp"},
            "message_count": {"$sum": 1},
            "first_message": {"$first": "$content"},
        }},
        {"$sort": {"last_timestamp": -1}},
        {"$project": {"_id": 0, "chat_session_id": "$_id", "last_message": 1,
                       "last_timestamp": 1, "message_count": 1, "first_message": 1}},
    ]
    return await db.chat_messages.aggregate(pipeline).to_list(100)

@api_router.delete("/chat/sessions/{chat_session_id}")
async def delete_chat_session(chat_session_id: str, request: Request):
    user_id = await get_current_user(request)
    result = await db.chat_messages.delete_many({"user_id": user_id, "chat_session_id": chat_session_id})
    return {"deleted": result.deleted_count}

# ============ Meal Planning ============

@api_router.post("/meals/generate")
async def generate_meal_plan(request: Request):
    user_id = await get_current_user(request)
    profile_context = await get_user_profile_context(user_id)
    import json
    try:
        chat = LlmChat(
            api_key=LLM_API_KEY,
            session_id=f"meal_{uuid.uuid4().hex[:8]}",
            system_message=f"""You are a nutrition expert. Generate a balanced daily meal plan.
{profile_context}
Return ONLY valid JSON (no markdown fences) as an array of meals with keys: meal_type, name, description, calories, protein, carbs, fats."""
        ).with_model("gemini", "gemini-2.0-flash")
        response = await chat.send_message(UserMessage(text="Generate a personalized healthy meal plan for today"))
        clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        meals = json.loads(clean)
    except Exception as e:
        logger.error(f"Meal parse error: {e}")
        meals = [
            {"meal_type": "Breakfast", "name": "Oatmeal with Berries & Almonds", "description": "Steel-cut oats topped with mixed berries", "calories": 350, "protein": 12, "carbs": 55, "fats": 10},
            {"meal_type": "Lunch", "name": "Grilled Chicken Quinoa Bowl", "description": "Grilled chicken with quinoa and vegetables", "calories": 520, "protein": 40, "carbs": 50, "fats": 16},
            {"meal_type": "Dinner", "name": "Baked Salmon with Sweet Potato", "description": "Salmon fillet with roasted sweet potato and broccoli", "calories": 480, "protein": 35, "carbs": 40, "fats": 18},
        ]
    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    doc = {"plan_id": plan_id, "user_id": user_id, "meals": meals,
           "date": datetime.now(timezone.utc).date().isoformat(),
           "created_at": datetime.now(timezone.utc).isoformat()}
    await db.meal_plans.insert_one(doc)
    return {"plan_id": plan_id, "meals": meals, "date": doc["date"]}

@api_router.get("/meals")
async def get_meal_plans(request: Request):
    user_id = await get_current_user(request)
    return await db.meal_plans.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(10).to_list(10)

@api_router.get("/meals/{plan_id}/pdf")
async def export_meal_pdf(plan_id: str, request: Request):
    user_id = await get_current_user(request)
    plan = await db.meal_plans.find_one({"plan_id": plan_id, "user_id": user_id}, {"_id": 0})
    if not plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    user_name = user_doc.get("name", "User") if user_doc else "User"
    return StreamingResponse(io.BytesIO(generate_meal_pdf(plan, user_name)),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=meal_plan_{plan.get('date', 'today')}.pdf"})

# ============ Workout Endpoints ============

@api_router.post("/workouts/generate")
async def generate_workout(request: Request):
    user_id = await get_current_user(request)
    profile_context = await get_user_profile_context(user_id)
    import json
    try:
        chat = LlmChat(
            api_key=LLM_API_KEY,
            session_id=f"workout_{uuid.uuid4().hex[:8]}",
            system_message=f"""You are a certified personal trainer. Generate a workout plan.
{profile_context}
Return ONLY valid JSON (no markdown fences) as an array of exercises with keys: name, sets, reps, rest, muscle_group, tips. Include 5-7 exercises."""
        ).with_model("gemini", "gemini-2.0-flash")
        response = await chat.send_message(UserMessage(text="Generate a personalized workout for today"))
        clean = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        exercises = json.loads(clean)
    except Exception as e:
        logger.error(f"Workout parse error: {e}")
        exercises = [
            {"name": "Barbell Squats", "sets": 4, "reps": "10", "rest": "90s", "muscle_group": "Legs", "tips": "Keep knees over toes"},
            {"name": "Push-ups", "sets": 3, "reps": "15", "rest": "60s", "muscle_group": "Chest", "tips": "Full range of motion"},
            {"name": "Bent-over Rows", "sets": 3, "reps": "12", "rest": "60s", "muscle_group": "Back", "tips": "Squeeze shoulder blades"},
            {"name": "Plank Hold", "sets": 3, "reps": "45s", "rest": "30s", "muscle_group": "Core", "tips": "Keep body straight"},
        ]
    workout_id = f"workout_{uuid.uuid4().hex[:12]}"
    doc = {"workout_id": workout_id, "user_id": user_id, "exercises": exercises,
           "date": datetime.now(timezone.utc).date().isoformat(), "completed": False,
           "created_at": datetime.now(timezone.utc).isoformat()}
    await db.workouts.insert_one(doc)
    return {"workout_id": workout_id, "exercises": exercises, "date": doc["date"]}

@api_router.get("/workouts")
async def get_workouts(request: Request):
    user_id = await get_current_user(request)
    return await db.workouts.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(10).to_list(10)

@api_router.put("/workouts/{workout_id}/complete")
async def complete_workout(workout_id: str, request: Request):
    user_id = await get_current_user(request)
    result = await db.workouts.update_one(
        {"workout_id": workout_id, "user_id": user_id}, {"$set": {"completed": True}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Workout not found")
    return {"message": "Workout completed"}

@api_router.get("/workouts/{workout_id}/pdf")
async def export_workout_pdf(workout_id: str, request: Request):
    user_id = await get_current_user(request)
    workout = await db.workouts.find_one({"workout_id": workout_id, "user_id": user_id}, {"_id": 0})
    if not workout:
        raise HTTPException(status_code=404, detail="Workout not found")
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    user_name = user_doc.get("name", "User") if user_doc else "User"
    return StreamingResponse(io.BytesIO(generate_workout_pdf(workout, user_name)),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=workout_{workout.get('date', 'today')}.pdf"})

# ============ Progress Endpoints ============

@api_router.post("/progress")
async def add_progress(entry: ProgressEntry, request: Request):
    user_id = await get_current_user(request)
    metric_id = f"metric_{uuid.uuid4().hex[:12]}"
    doc = {"metric_id": metric_id, "user_id": user_id,
           "weight": entry.weight, "bmi": entry.bmi, "body_fat": entry.body_fat,
           "muscle_mass": entry.muscle_mass, "calories_burned": entry.calories_burned,
           "workout_minutes": entry.workout_minutes,
           "date": datetime.now(timezone.utc).date().isoformat(),
           "created_at": datetime.now(timezone.utc).isoformat()}
    await db.progress_metrics.insert_one(doc)
    return {"metric_id": metric_id}

@api_router.get("/progress")
async def get_progress(request: Request):
    user_id = await get_current_user(request)
    return await db.progress_metrics.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(30).to_list(30)

@api_router.get("/progress/summary")
async def get_progress_summary(request: Request):
    user_id = await get_current_user(request)
    metrics = await db.progress_metrics.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(30).to_list(30)
    if not metrics:
        return {"total_entries": 0, "latest": None, "trend": None}
    latest = metrics[0]
    oldest = metrics[-1] if len(metrics) > 1 else None
    trend = None
    if oldest and latest.get("weight") and oldest.get("weight"):
        diff = latest["weight"] - oldest["weight"]
        trend = "gaining" if diff > 0 else "losing" if diff < 0 else "maintaining"
    total_workouts = await db.workouts.count_documents({"user_id": user_id, "completed": True})
    total_meals = await db.meal_plans.count_documents({"user_id": user_id})
    return {"total_entries": len(metrics), "latest": latest, "trend": trend,
            "total_workouts_completed": total_workouts, "total_meal_plans": total_meals}

# ============ RAG Cache Endpoints ============

@api_router.post("/rag/refresh")
async def refresh_rag_cache(request: Request):
    await get_current_user(request)
    results = {}
    for intent in ["WORKOUT", "NUTRITION", "DIETARY_ADVICE", "GENERAL_HEALTH"]:
        await db.scrape_cache.delete_one({"cache_key": f"scrape_cache_{intent}"})
        content = await get_scraped_knowledge(intent)
        results[intent] = len(content) > 0
    return {"refreshed": results}

@api_router.get("/rag/status")
async def get_rag_status(request: Request):
    await get_current_user(request)
    statuses = []
    for intent, sources in SCRAPE_SOURCES.items():
        cached = await db.scrape_cache.find_one({"cache_key": f"scrape_cache_{intent}"}, {"_id": 0})
        statuses.append({"intent": intent, "sources": [s["name"] for s in sources],
                          "cached": cached is not None,
                          "cached_at": cached.get("cached_at") if cached else None,
                          "content_length": len(cached.get("content", "")) if cached else 0})
    return {"knowledge_sources": statuses}

# ============ Register router — ONCE, at the end ============
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)