from datetime import datetime

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


async def send_team_access_email(
    to_email: str, member_name: str, owner_name: str, verify_link: str
) -> None:
    """Send a verification email to a newly added team member."""
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">You've been added to a team</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{member_name}</strong>, <strong>{owner_name}</strong> has given you access to their dashboard.</p>
        <p style="color:#374151;margin-bottom:16px;">Click the button below to verify your email and activate your access:</p>
        <a href="{verify_link}"
           style="display:inline-block;background:#111827;color:#fff;text-decoration:none;padding:12px 28px;border-radius:8px;font-size:14px;font-weight:600;">
            Verify &amp; Activate Access
        </a>
        <p style="color:#6b7280;font-size:13px;margin-top:24px;">Or copy this link:<br>
            <span style="color:#374151;word-break:break-all;">{verify_link}</span>
        </p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">If you weren't expecting this, you can safely ignore this email.</p>
    </div>
    """
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json={
                "from": settings.RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": f"You've been added to {owner_name}'s team",
                "html": html_body,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Resend API error {response.status_code}: {response.text}")


async def send_team_access_otp_email(to_email: str, member_name: str, otp: str) -> None:
    """Send a sign-in OTP to a team member (who has no user account)."""
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Team Sign-In Code</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{member_name}</strong>, use the code below to sign in to your team dashboard.</p>
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
                "subject": "Your Team Sign-In Code",
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


async def send_subscription_confirmed_email(
    to_email: str,
    company_name: str,
    tier: str,
    billing_cycle: str,
    amount: float,
    currency: str,
    period_end: datetime | None = None,
) -> None:
    """Sent once when a paid subscription becomes active (Stripe checkout.session.completed)."""

    subject = f"Your {tier.title()} plan is active"
    amount_str = f"{amount:,.2f} {currency.upper()}" if amount else "$0.00"
    renews_line = (
        f"<p style=\"color:#6b7280;font-size:13px;margin-top:12px;\">Renews on <strong>{period_end.strftime('%B %d, %Y')}</strong>.</p>"
        if period_end else ""
    )

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Subscription confirmed</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, your subscription is now active — thanks for signing up!</p>

        <div style="background:#f3f4f6;border-radius:8px;padding:20px;margin-bottom:24px;">
            <p style="color:#374151;margin:0 0 6px;"><strong>Plan:</strong> {tier.title()} ({billing_cycle})</p>
            <p style="color:#374151;margin:0;"><strong>Amount:</strong> {amount_str} / {billing_cycle}</p>
        </div>

        <p style="color:#6b7280;font-size:13px;">Your AI assistant is ready to go — head to your dashboard to train it and start capturing leads.</p>
        {renews_line}
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">Questions about your billing? Reach us at info@goconverto.com.</p>
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


async def send_subscription_ending_soon_email(
    to_email: str, company_name: str, ends_at: datetime | None,
) -> None:
    """Sent ~1 day before a canceled-but-still-active subscription's access actually ends."""

    when = ends_at.strftime("%B %d, %Y") if ends_at else "tomorrow"
    subject = "Your subscription ends tomorrow"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Your access ends tomorrow</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, your subscription is set to cancel and your access ends on <strong>{when}</strong>.</p>

        <p style="color:#374151;margin-bottom:8px;">Want to keep your AI assistant running? You can resubscribe anytime before then from your billing settings — no need to set anything up again.</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">If you meant to cancel, no action is needed — your account will move to the free plan automatically.</p>
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


async def send_subscription_ended_email(to_email: str, company_name: str) -> None:
    """Sent when a subscription has fully ended (Stripe customer.subscription.deleted)."""

    subject = "Your subscription has ended"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Subscription ended</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, your subscription has ended and your account is now on the free plan.</p>

        <p style="color:#374151;margin-bottom:8px;">You can resubscribe at any time to restore full access for your AI assistant.</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">Questions? Reach us at info@goconverto.com.</p>
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
