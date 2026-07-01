import httpx

from config import settings

RESEND_API_URL = "https://api.resend.com/emails"


async def send_otp_email(to_email: str, company_name: str, otp: str) -> None:
    """Send a 6-digit OTP verification email via the Resend API."""

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

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json={
                "from": settings.RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Resend API error {response.status_code}: {response.text}")


async def send_invite_email(
    to_email: str, invitee_name: str, owner_name: str, invite_link: str
) -> None:
    """Send a team access invite email via the Resend API."""

    subject = f"You've been invited to join {owner_name}"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">You've been invited</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{invitee_name}</strong>, <strong>{owner_name}</strong> has invited you to access their dashboard.</p>

        <a href="{invite_link}"
           style="display:inline-block;background:#111827;color:#fff;text-decoration:none;padding:12px 24px;border-radius:8px;font-size:14px;font-weight:600;">
            Accept Invite
        </a>

        <p style="color:#6b7280;font-size:13px;margin-top:24px;">
            Or copy this link into your browser:<br>
            <span style="color:#374151;word-break:break-all;">{invite_link}</span>
        </p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">If you weren't expecting this invite, you can safely ignore this email.</p>
    </div>
    """

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json={
                "from": settings.RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Resend API error {response.status_code}: {response.text}")


async def send_login_otp_email(to_email: str, company_name: str, otp: str) -> None:
    """Send a 6-digit OTP for passwordless sign-in via the Resend API."""

    subject = "Your Sign-In Code"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Sign-In Verification</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, use the code below to sign in.</p>

        <p style="color:#374151;margin-bottom:8px;">Your one-time sign-in code is:</p>
        <div style="background:#f3f4f6;border-radius:8px;padding:20px;text-align:center;margin-bottom:24px;">
            <span style="font-size:36px;font-weight:bold;letter-spacing:12px;color:#111827;">{otp}</span>
        </div>

        <p style="color:#6b7280;font-size:13px;">This code expires in <strong>10 minutes</strong>. Do not share it with anyone.</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">If you didn't request this, you can safely ignore this email.</p>
    </div>
    """

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json={
                "from": settings.RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Resend API error {response.status_code}: {response.text}")
