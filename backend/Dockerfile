# Docker image ke liye base Python version choose karein.
# Ham Python 3.10-slim use kar rahe hain, kyunke yeh stable aur lightweight hai.
# Agar aapko audioop-lts wapis chahiye aur use Python 3.13+ ki zaroorat hai,
# to yahan 'python:3.13-slim' ya 'python:3.13-alpine' try kar sakte hain,
# lekin pehle 3.10 ya 3.11 par hi deploy karne ki koshish karein.
FROM python:3.13-slim

# Container ke andar kaam karne ki directory set karein.
# Saara code isi directory mein hoga.
WORKDIR /app

# 'requirements.txt' file ko apne project se container mein copy karein.
# Yeh file aapki saari Python libraries ki list par mushtamil hoti hai.
COPY requirements.txt .

# 'requirements.txt' mein di gayi saari Python libraries install karein.
# '--no-cache-dir' installation ko thora fast karta hai aur space bachata hai.
RUN pip install --no-cache-dir -r requirements.txt

# Aapke baqi saare code files (jaisa ke main.py, FAISS files, etc.) ko
# project se container ki /app directory mein copy karein.
COPY . .

# Jab container start hoga, to yeh command chalega jo aapki FastAPI app ko run karega.
# "main:app" - "main" aapki FastAPI app ki file ka naam hai (misal ke taur par main.py).
#              "app" us main.py file ke andar aapke FastAPI instance ka naam hai (misal ke taur par app = FastAPI()).
# "--host 0.0.0.0" - app ko har network interface par accessible banata hai.
# "--port 7860" - Hugging Face Spaces aam taur par apps ko port 7860 par expose karte hain.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]