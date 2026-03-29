from fastapi import APIRouter, HTTPException, status, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_database
from schemas.user import SignupRequest, OTPVerifyRequest, SigninRequest, TokenResponse
from services import auth_service

router = APIRouter(prefix="/auth", tags=["Auth"])

ERROR_MAP = {
    "email_taken":         (status.HTTP_409_CONFLICT,       "Email is already registered."),
    "email_send_failed":   (status.HTTP_502_BAD_GATEWAY,    "Failed to send OTP email. Try again."),
    "otp_not_found":       (status.HTTP_404_NOT_FOUND,      "No pending signup found for this email."),
    "invalid_otp":         (status.HTTP_400_BAD_REQUEST,    "Invalid OTP code."),
    "otp_expired":         (status.HTTP_400_BAD_REQUEST,    "OTP has expired. Please sign up again."),
    "invalid_credentials": (status.HTTP_401_UNAUTHORIZED,   "Invalid email or password."),
    "email_not_verified":  (status.HTTP_403_FORBIDDEN,      "Please verify your email before signing in."),
    "account_disabled":    (status.HTTP_403_FORBIDDEN,      "Your account has been disabled."),
}


def raise_if_error(result: dict):
    error = result.get("error")
    if error:
        code, detail = ERROR_MAP.get(error, (status.HTTP_400_BAD_REQUEST, error))
        raise HTTPException(status_code=code, detail=detail)


@router.post("/signup", status_code=status.HTTP_201_CREATED, summary="Register company & send OTP")
async def signup(payload: SignupRequest, db: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Step 1 — Submit company details.
    An OTP is sent to the provided email. Account is created only after verification.
    """
    result = await auth_service.signup(db, payload)
    raise_if_error(result)
    return result


@router.post("/verify-otp", summary="Confirm OTP and create account")
async def verify_otp(payload: OTPVerifyRequest, db: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Step 2 — Enter the 6-digit OTP received by email.
    On success the account is created and ready to use.
    """
    result = await auth_service.verify_otp(db, payload)
    raise_if_error(result)
    return result


@router.post("/signin", response_model=TokenResponse, summary="Sign in and get JWT token")
async def signin(payload: SigninRequest, db: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Sign in with email + password.
    Returns a JWT bearer token valid for use in all protected endpoints.
    """
    result = await auth_service.signin(db, payload)
    raise_if_error(result)
    return result
