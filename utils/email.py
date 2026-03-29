import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import settings


async def send_otp_email(to_email: str, company_name: str, otp: str) -> None:
    """Send a 6-digit OTP verification email via Gmail SMTP SSL."""

    subject = "Verify Your Email — OTP Code"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Email Verification</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, thank you for signing up!</p>

        <p style="color:#374151;margin-bottom:8px;">Your one-time verification code is:</p>
        <div style="background:#f3f4f6;border-radius:8px;padding:20px;text-align:center;margin-bottom:24px;">
            <span style="font-size:36px;font-weight:bold;letter-spacing:12px;color:#111827;">{otp}</span>
        </div>

        <p style="color:#6b7280;font-size:13px;">This code expires in <strong>10 minutes</strong>. Do not share it with anyone.</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">If you didn't request this, you can safely ignore this email.</p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_MAIL
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    await aiosmtplib.send(
        msg,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_MAIL,
        password=settings.SMTP_PASSWORD,
        use_tls=True,           # port 465 = SSL (use_tls=True)
    )
