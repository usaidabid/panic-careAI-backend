import os
# requests library ki ab zaroorat nahi Twilio ke liye, isko hata diya
from dotenv import load_dotenv
from twilio.rest import Client # Twilio ki library import ki

# Environment variables load karein .env file se
load_dotenv()

# --- Twilio Credentials ---
# Yeh credentials aapko apne .env file mein set karne honge.
# Example in .env:
# TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TWILIO_AUTH_TOKEN=your_auth_token_here
# TWILIO_PHONE_NUMBER=+1234567890 (aapka Twilio number, jahan se SMS jayega)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Twilio client ko initialize karein
twilio_client = None
twilio_configured = False

# Check karein ke Twilio credentials properly set hain
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twilio_configured = True
    print("Twilio client configured successfully.")
else:
    print("Warning: Twilio credentials (ACCOUNT_SID, AUTH_TOKEN, or PHONE_NUMBER) are missing/dummy. SMS sending will be mocked.")

def send_sms(to_phone_number: str, message: str) -> bool:
    """
    Sends an SMS message using Twilio, or mocks the sending if credentials are not fully set.
    Returns True if successful (or mocked), False otherwise.
    """
    # Agar Twilio set nahi hai, to mocked SMS send karein
    if not twilio_configured:
        print(f"MOCK SMS SENT to {to_phone_number} (via Twilio mock): '{message}'")
        print("Note: This is a mocked SMS because Twilio client is not fully configured.")
        return True # Testing purposes ke liye mocked send ko successful mana jayega

    # --- E.164 Format Validation and Auto-formatting ---
    # Twilio ko bhi phone numbers E.164 format mein chahiye (e.g., +923001234567).
    # Yeh logic 03xxxxxxxxx, 923xxxxxxxxx, aur +923xxxxxxxxx formats ko handle karegi Pakistan ke liye.
    
    e164_formatted_number = to_phone_number # Original number se initialize karein

    # Agar number '+' se shuru nahi ho raha, to format karne ki koshish karein
    if not to_phone_number.startswith('+'):
        if to_phone_number.startswith('0') and len(to_phone_number) == 11:
            # Format: 03001234567 -> +923001234567
            e164_formatted_number = '+92' + to_phone_number[1:]
        elif to_phone_number.startswith('92') and len(to_phone_number) == 12:
            # Format: 923001234567 -> +923001234567 (seedha '+' lagayen)
            e164_formatted_number = '+' + to_phone_number
        elif len(to_phone_number) == 10 and to_phone_number.startswith('3'):
            # Format: 3001234567 -> +923001234567 (agar 0 ya 92 prefix nahi hai)
            e164_formatted_number = '+92' + to_phone_number
        else:
            # Baqi unexpected formats ke liye - is se masle ho sakte hain
            print(f"Warning: Destination number '{to_phone_number}' does not match common Pakistani E.164 formats (03xxxxxxxxx, 923xxxxxxxxx, +923xxxxxxxxx). Attempting to prepend '+' as a fallback.")
            e164_formatted_number = '+' + to_phone_number 
    
    # Formatted number ki final validation
    if not (e164_formatted_number.startswith('+') and len(e164_formatted_number) >= 12 and e164_formatted_number[1:].isdigit()):
        print(f"Error: Final formatted number '{e164_formatted_number}' is invalid or too short after formatting. Cannot send SMS.")
        return False

    # --- Send SMS via Twilio API ---
    try:
        # Twilio messages create method use karein
        message_response = twilio_client.messages.create(
            to=e164_formatted_number,
            from_=TWILIO_PHONE_NUMBER, # Yahan Twilio ka apna phone number aayega
            body=message # SMS message ka content
        )

        # Twilio API call successful hone par Message SID mil jata hai.
        # Delivery status Twilio dashboard logs mein check kar sakte hain.
        print(f"SMS send request accepted by Twilio for {e164_formatted_number}. Message SID: {message_response.sid}.")
        print("Check Twilio dashboard logs for final delivery status.")
        return True

    except Exception as e:
        # Koi bhi error hone par catch karein
        print(f"Failed to send SMS via Twilio to {e164_formatted_number}: {e}")
        print("Please ensure Twilio credentials, phone number are correct, and check network connectivity.")
        return False