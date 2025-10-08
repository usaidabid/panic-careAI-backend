from fastapi import FastAPI, Request, UploadFile, File, status, Depends, HTTPException, Form
from typing import Optional, Union
from fastapi.responses import StreamingResponse
from fastapi import Response
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from backend.rag_pipeline import qa_chain  # Import your RetrievalQA chain
import speech_recognition as sr
import io
from gtts import gTTS
import re
import tempfile
import os
from dotenv import load_dotenv
from fastapi.security import OAuth2PasswordRequestForm
import whisper
import soundfile as sf
import torch
import numpy as np
from whisper import audio as audio_utils
from whisper import pad_or_trim, log_mel_spectrogram, DecodingOptions, decode # <--- NEW
import wave
import mimetypes


load_dotenv()
from backend.database import database, create_db_and_tables, SessionLocal, User
from backend.security import get_password_hash, verify_password
from backend.auth import create_access_token, get_current_username, Token
from backend.notification_service import send_sms



os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin"




whisper_model = whisper.load_model("base") 


app = FastAPI(
    title="Panic Care AI Backend",
    description="Backend for Panic Care AI application with user authentication and emergency notifications.",
    version="1.0.0",
)
 
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:60000",   # tumhara frontend origin
    "http://127.0.0.1:60000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://725489125ee0.ngrok-free.app" ,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ab specific origins list use ho rahi hai
    allow_credentials=False,  # login/register ke liye zaroori hai
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Startup/Shutdown Events ---
@app.on_event("startup")
async def startup():
    """Connects to the database and creates tables on application startup."""
    await database.connect()
    create_db_and_tables() # Ensure tables are created
    print("Database connected and tables created.")

@app.on_event("shutdown")
async def shutdown():
    """Disconnects from the database on application shutdown."""
    await database.disconnect()
    print("Database disconnected.")

# Dependency to get DB session for each request
def get_db():
    """Provides a SQLAlchemy database session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models ---

# Input format for general text queries
class QueryRequest(BaseModel):
    question: str

# Output format for RAG answers
class AnswerResponse(BaseModel):
    answer: str

# User related models
class UserCreate(BaseModel):
    username: str
    password: str
    doctor_whatsapp_number: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str




class UserResponse(BaseModel):
    id: int
    username: str
    doctor_whatsapp_number: Optional[str] = None

    class Config:
        orm_mode = True # For Pydantic V1. For Pydantic V2+, use: from_attributes = True

# --- NEW: Model for Emergency Call Request (for "Call Doctor" button) ---
class EmergencyCallRequest(BaseModel):
    doctor_phone_number: str
    user_latitude: float
    user_longitude: float
    emergency_brief: Optional[str] = None # Optional: User ki current state ka brief



# --- User Authentication Endpoints ---

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Registers a new user account.
    Checks if username already exists.
    Hashes the password and stores user data, including doctor's WhatsApp number.
    Sends an initial notification to the doctor if a number is provided.
    """
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    hashed_password = get_password_hash(user.password)

    new_user = User(
        username=user.username,
        hashed_password=hashed_password,
        doctor_whatsapp_number=user.doctor_whatsapp_number
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    if new_user.doctor_whatsapp_number:
        message_to_doctor = (
            f"Dear Doctor, a new patient '{new_user.username}' has registered "
            f"with your number in the Panic Care AI app. "
            f"Please be aware of potential emergency notifications from this patient."
        )
        send_sms(new_user.doctor_whatsapp_number, message_to_doctor)

    return new_user

@app.post("/token", response_model=Token, tags=["Users"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Authenticates a user and issues an access token upon successful login.
    """
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.username}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse, tags=["Users"])
async def read_users_me(
    current_username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Retrieves the details of the currently authenticated user.
    Requires a valid JWT access token.
    """
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# --- RAG/Chatbot Endpoints ---

@app.post("/query", response_model=AnswerResponse, tags=["RAG"])
async def get_answer(
    request: QueryRequest,
    current_username: str = Depends(get_current_username)
):
    """
    Processes a text query using the RAG system.
    Requires authentication.
    """
    print(f"Query from user '{current_username}': {request.question}")

    question = request.question
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    return {"answer": answer}

"""@app.post("/transcribe_and_speak", tags=["RAG"])
async def transcribe_and_speak(
    # CHANGE 1: `audio_file` type hint ko `Union[UploadFile, None]` kiya gaya hai.
    audio_file: Union[UploadFile, None] = File(None),
    # REMOVED: query_text parameter ko yahan se hata diya gaya hai.
    
    # CHANGE 2: `latitude` aur `longitude` parameters ko hata diya gaya hai.
    # Ab inka koi maqsad nahi hai is endpoint mein.
    # latitude: Optional[float] = Form(None),
    # longitude: Optional[float] = Form(None),

    current_username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
     
    # Step 1: Authenticated user ki details fetch karna
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="Authenticated user not found.")

    transcribed_text = None
    # CHANGE 3: `audio_file` ko empty string se handle kiya gaya hai.
    # Agar Swagger UI empty string bhejta hai, to usay None samjha jaye.
    if audio_file == "" or (isinstance(audio_file, UploadFile) and not audio_file.filename):
        audio_file = None

    # Step 2: Audio Transcription
    if audio_file: # Ab yeh check correctly None ko handle karega.
        try:
            audio_content = await audio_file.read()

            import tempfile, os
            orig_name = getattr(audio_file, "filename", None) or "upload.wav"
            _, ext = os.path.splitext(orig_name)
            if not ext:
                ext = ".wav"

            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f"upload_{os.getpid()}{ext}")

            with open(tmp_path, "wb") as f:
                f.write(audio_content)


            result = whisper_model.transcribe(tmp_path, language="en")
            transcribed_text = result["text"]

            print(f"Whisper Transcription: {transcribed_text}")
            os.remove(tmp_path)

        except Exception as e:
            print(f"Whisper Error: {e}")
            transcribed_text = "Sorry, I could not process your voice input."

            
    
    # Step 3: Final query text determine karna aur us par robust normalization apply karna.
    # IMPORTANT CHANGE: Ab yahan sirf transcribed_text se hi final_query banega.
    final_query = transcribed_text 
    
    if final_query:
        # CHANGE 5 & 6: Robust string normalization steps.
        # Ye hidden characters aur extra spaces ko remove kar ke query ko clean karega.
        final_query = final_query.strip() # Shuru aur aakhir se whitespace hatana
        final_query = final_query.lower() # Sab kuch lowercase karna
        # Non-alphanumeric characters (jaise '?', '!', commas, periods) remove karna
        final_query = re.sub(r'[^a-z0-9\s]', '', final_query) 
        # Multiple spaces ko single space mein badalna
        final_query = re.sub(r'\s+', ' ', final_query)
        final_query = final_query.strip() # Dobara strip karna, in case new spaces add hue hon

    print(f"Final Query for RAG (normalized repr): {repr(final_query)}") # Normalized query ko print karna


    if not final_query:
        # IMPORTANT CHANGE: Error message ko update kiya gaya hai.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio input provided for processing. Please provide an audio file."
        )

    # Step 4: Query ko RAG/LangChain system ke through process karna
    answer_text = ""
    try:
        # Assuming `qa_chain` correctly configured aur globally available hai
        result = qa_chain.invoke({"query": final_query})
        answer_text = result["result"]
        print(f"RAG Answer Text: {answer_text[:100]}...")
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        answer_text = "I encountered an issue while processing your query with the RAG system."

    # Step 5: Text-to-Speech (TTS) mein convert karna
    tts_response_message = answer_text
    if not tts_response_message:
        tts_response_message = "Sorry, I couldn't process your request fully. Please try again."

    tts = gTTS(text=tts_response_message, lang='en', slow=False) # gTTS object
    audio_bytes_io = io.BytesIO() # Audio data store karne ke liye
    tts.write_to_fp(audio_bytes_io) # TTS output ko BytesIO stream mein likhna
    
    print(f"Generated audio size: {audio_bytes_io.tell()} bytes") # Generate kiye gaye audio ka size

    audio_bytes_io.seek(0) # Stream ko shuruat mein lana taake read ho sake
    
    # --- YEH HAIN WO ZAROORI CHANGES JO `Illegal header value` ERROR KO THEEK KARENGE ---
    # `answer_text` ko clean aur chota kar ke `X-RAG-Response` header mein bhejein.
    # New lines ko spaces se replace karein.
    cleaned_answer_for_header = answer_text.replace('\n', ' ').replace('\r', '') # New lines remove
    
    # Agar answer bohat lamba hai, to usay truncate karein (maslan pehle 200 characters).
    # HTTP headers mein bohat lambi values allowed nahi hoti.
    if len(cleaned_answer_for_header) > 200:
        cleaned_answer_for_header = cleaned_answer_for_header[:197] + "..." # ... add karein
    
    headers = {
        "X-RAG-Response": cleaned_answer_for_header, # Ab yehi value jayegi
        #"Content-Disposition": "attachment; filename=response.mp3" # Browser download option dega
    }
    return StreamingResponse(
    io.BytesIO(audio_bytes_io.getvalue()),
    media_type="audio/mpeg",
    headers=headers
)"""


@app.post("/transcribe_and_speak", tags=["RAG"])
async def transcribe_and_speak(
    audio_file: Union[UploadFile, None] = File(None),
    current_username: str = Depends(get_current_username),
    db: Session = Depends(get_db)
):
    """
    Handles audio queries from authenticated users, processing them
    through the RAG system and returning a spoken (TTS) response.
    """

    # Step 1: Authenticated user fetch
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="Authenticated user not found.")

    transcribed_text = None

    # Handle empty audio input
    if audio_file == "" or (isinstance(audio_file, UploadFile) and not audio_file.filename):
        audio_file = None

    # ✅ Step 2: Audio Transcription
    if audio_file:
        try:
            # Read uploaded bytes
            audio_content = await audio_file.read()

            # Detect MIME type and extension (mostly 'audio/wav')
            mime_type = audio_file.content_type or "audio/wav"
            ext = ".wav" if "wav" in mime_type else mimetypes.guess_extension(mime_type) or ".wav"

            # Save received audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(audio_content)
                tmp.flush()
                tmp_path = tmp.name

            print(f"🎙️ Received audio: {tmp_path} ({mime_type}) — {len(audio_content)} bytes")

            # ✅ Since Flutter already sends WAV (PCM 16kHz), we skip FFmpeg conversion
            safe_wav_path = tmp_path

            # Quick validation to ensure it's readable WAV
            try:
                with wave.open(safe_wav_path, 'rb') as wav_file:
                    channels = wav_file.getnchannels()
                    rate = wav_file.getframerate()
                    print(f"✅ Valid WAV file — Channels: {channels}, Sample Rate: {rate}")
            except Exception as e:
                print(f"⚠️ WAV validation failed: {e}")
                transcribed_text = "Audio file invalid or corrupted."
                os.remove(tmp_path)
                return {"error": transcribed_text}

            # ✅ Whisper Transcription
            result = whisper_model.transcribe(safe_wav_path, language="en", fp16=False)
            transcribed_text = result.get("text", "").strip()
            print(f"🎤 User said: {transcribed_text}")

            # Clean up
            os.remove(tmp_path)

        except Exception as e:
            print(f"❌ Whisper Error: {e}")
            transcribed_text = "Sorry, I could not process your voice input."
    else:
        raise HTTPException(status_code=400, detail="No audio input provided.")

    # Step 3: Normalize and clean query
    final_query = transcribed_text
    if final_query:
        final_query = re.sub(r'[^a-z0-9\s]', '', final_query.lower()).strip()

    print(f"Final Query for RAG (normalized repr): {repr(final_query)}")

    if not final_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid speech content for processing."
        )

    # Step 4: Query through RAG chain
    answer_text = ""
    try:
        result = qa_chain.invoke({"query": final_query})
        answer_text = result["result"]
        print(f"RAG Answer Text: {answer_text[:100]}...")
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        answer_text = "I encountered an issue while processing your query."

    # Step 5: Text-to-Speech (TTS)
    tts_response_message = answer_text or "Sorry, I couldn't process your request fully."
    tts = gTTS(text=tts_response_message, lang='en', slow=False)
    audio_bytes_io = io.BytesIO()
    tts.write_to_fp(audio_bytes_io)
    audio_bytes_io.seek(0)
    print(f"Generated audio size: {audio_bytes_io.tell()} bytes")

    # Step 6: Clean header
    cleaned_header = answer_text.replace("\n", " ").replace("\r", "")
    if len(cleaned_header) > 200:
        cleaned_header = cleaned_header[:197] + "..."

    headers = {"X-RAG-Response": cleaned_header}

    return StreamingResponse(
        audio_bytes_io,
        media_type="audio/mpeg",
        headers=headers
    )
    #return StreamingResponse(audio_bytes_io, media_type="audio/mpeg", headers=headers)
# --- NEW: Emergency Call Endpoint for "Call Now" Button ---
@app.post("/initiate-emergency-call", status_code=status.HTTP_200_OK, tags=["Emergency"])
async def initiate_emergency_call(
    request: EmergencyCallRequest,
    current_username: str = Depends(get_current_username), # User authentication required
    db: Session = Depends(get_db)
):
    """
    Sends patient's location and an emergency brief
    to the linked doctor via SMS and prepares for native dialer call.
    """
    print(f"Emergency brief initiated by user: {current_username}")
    print(f"Target Doctor: {request.doctor_phone_number}")
    print(f"User Location: Lat {request.user_latitude}, Lon {request.user_longitude}")
    print(f"Emergency Brief: {request.emergency_brief if request.emergency_brief else 'No specific brief provided.'}")

    # Optional: Yahan aap verify kar sakte hain ke kya frontend se aane wala doctor_phone_number
    # user ke registered doctor_whatsapp_number se match karta hai.
    # Agar yeh check zaroori hai, to uncomment karein aur isko implement karein.
    # user = db.query(User).filter(User.username == current_username).first()
    # if not user or user.doctor_whatsapp_number != request.doctor_phone_number:
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor number mismatch or user not found.")


    try:
        # 1. Google Maps link banao
        # Note: '2' in the URL is just an arbitrary number for pathing purposes, 
        # actual Google Maps links dynamically format.
        location_link = f"http://maps.google.com/maps?q={request.user_latitude},{request.user_longitude}" 
        
        # 2. SMS message for doctor
        message_to_doctor = (
            f"EMERGENCY ALERT from PanicCareAI! Patient '{current_username}' needs immediate help. "
            f"Condition: {request.emergency_brief if request.emergency_brief else 'No specific brief provided.'}. "
            f"Patient Location: {location_link}. "
            f"Please check on them and call them back if needed."
        )
        
        # 3. Infobip SMS API se message send karo
        sms_sent = send_sms(request.doctor_phone_number, message_to_doctor)

        if not sms_sent:
            print("ERROR: Failed to send emergency SMS via twilio.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send emergency SMS. Please check server logs."
            )

        print("SUCCESS: Emergency location SMS sent to doctor.")
        
        return {
            "message": "Emergency SMS sent to doctor.",
            "location_link": location_link, # Frontend verify kar sakta hai
            "doctor_phone_number": request.doctor_phone_number # Frontend ke liye confirm number
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred in /initiate-emergency-call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred during emergency request. Please try again."
        )

    
@app.get("/account" , response_model = UserResponse , tags =["Users"])
async def show_user(
     current_username: str = Depends(get_current_username),
     db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == current_username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


        

# --- Health Check Endpoint (Optional but Recommended) ---
@app.get("/health", tags=["System"])
async def health_check():
    """Checks the health of the API."""
    return {"status": "ok", "message": "API is running"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)