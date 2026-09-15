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


async def send_plan_changed_email(
    to_email: str,
    company_name: str,
    tier: str,
    billing_cycle: str,
    amount_charged: float | None,
    currency: str,
    scheduled: bool,
    effective_at: datetime | None,
) -> None:
    """
    Sent from change_subscription_plan right after a plan switch completes —
    either charged immediately (upgrade / trial ending into a paid plan /
    same-price switch) or scheduled for the next renewal with no charge
    (downgrade). Only called once the outcome is definitively known (i.e.
    not while a payment confirmation is still pending on the frontend).
    """
    plan_label = f"{tier.title()} ({billing_cycle})"

    if scheduled:
        subject = f"Switching to {tier.title()} at your next renewal"
        when = effective_at.strftime("%B %d, %Y") if effective_at else "your next renewal date"
        body_line = (
            f"Your plan will switch to <strong>{plan_label}</strong> on <strong>{when}</strong>. "
            "You'll keep your current plan's features until then — no charge today, "
            "and no refund for the current period."
        )
    else:
        amount_str = f"${amount_charged:,.2f} {currency.upper()}" if amount_charged else "$0.00"
        subject = f"You're now on {tier.title()}"
        body_line = (
            f"You've switched to <strong>{plan_label}</strong> and <strong>{amount_str}</strong> "
            "was charged to your card on file just now."
        )

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Plan changed</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, this confirms your plan change.</p>

        <p style="color:#374151;margin-bottom:8px;">{body_line}</p>
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


async def send_cancellation_received_email(
    to_email: str, company_name: str, access_until: datetime | None, immediately: bool,
) -> None:
    """Sent synchronously right when cancel_subscription() succeeds — doesn't
    wait on a Stripe webhook, since immediately=False (the dashboard's
    default) never fires one that would otherwise confirm this to the
    customer until right before the period actually ends."""

    if immediately:
        subject = "Your subscription has been canceled"
        body_line = "Your subscription has been canceled immediately — you no longer have paid access."
    elif access_until:
        subject = "We've received your cancellation"
        body_line = (
            f"You'll keep full access until <strong>{access_until.strftime('%B %d, %Y')}</strong>, "
            "after which your account moves to the free plan. You won't be charged again."
        )
    else:
        subject = "We've received your cancellation"
        body_line = "Your subscription won't renew. You won't be charged again."

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Cancellation confirmed</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, this confirms your cancellation request.</p>

        <p style="color:#374151;margin-bottom:8px;">{body_line}</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">Changed your mind? You can resubscribe anytime from your billing settings.</p>
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


async def send_payment_due_email(
    to_email: str, company_name: str, grace_period_end: datetime | None,
) -> None:
    """Sent the moment a charge fails (post-trial or any renewal) — the
    subscription is now past_due. Grace period is still fully open at this
    point, but the copy must not imply there's no rush."""

    when = grace_period_end.strftime("%B %d, %Y") if grace_period_end else "in a few days"
    subject = "Action needed: your payment didn't go through"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Payment failed</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, we couldn't charge your card for your subscription.</p>

        <p style="color:#374151;margin-bottom:8px;">Please update your payment method by <strong>{when}</strong>. If payment isn't received by then, your chatbot will stop answering your visitors' questions until you pay.</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">You can update your payment method anytime from your billing settings.</p>
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


async def send_payment_hard_reminder_email(
    to_email: str, company_name: str, grace_period_end: datetime | None,
) -> None:
    """Sent ~3 days into the grace period — a stronger-worded follow-up to
    send_payment_due_email for a still-unpaid past_due subscription."""

    when = grace_period_end.strftime("%B %d, %Y") if grace_period_end else "in a few days"
    subject = "Final notice: your chatbot will be turned off soon"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #fca5a5;border-radius:8px;">
        <h2 style="color:#b91c1c;margin-bottom:4px;">Your chatbot will be turned off</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, we still haven't been able to charge your card.</p>

        <p style="color:#374151;margin-bottom:8px;">If payment isn't received by <strong>{when}</strong>, your chatbot will stop answering your visitors' questions. Update your payment method now to avoid any interruption.</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">You can update your payment method anytime from your billing settings.</p>
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


async def send_chatbot_overdue_email(to_email: str, company_name: str) -> None:
    """Sent once the 7-day grace period lapses with no payment — the
    chatbot has just been turned off for real."""

    subject = "Your chatbot has been paused"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Your chatbot is now paused</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, your subscription is overdue, so your chatbot has stopped answering visitor questions.</p>

        <p style="color:#374151;margin-bottom:8px;">Your dashboard and data are safe and nothing is lost. Pay now from your billing settings to turn your chatbot back on immediately.</p>
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


async def send_visitor_limit_reached_email(to_email: str, company_name: str) -> None:
    """Sent the first time a billing period's visitor cap is hit — once per
    period, not on every blocked visitor."""

    subject = "You've reached your visitor limit"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
        <h2 style="color:#1f2937;margin-bottom:4px;">Visitor limit reached</h2>
        <p style="color:#6b7280;margin-bottom:24px;">Hi <strong>{company_name}</strong>, your AI assistant has used up all the visitors included in your plan for this billing period.</p>

        <p style="color:#374151;margin-bottom:8px;">New visitors won't get AI replies until you upgrade your plan or your next billing period starts. Existing conversations already in progress aren't affected.</p>
        <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
        <p style="color:#9ca3af;font-size:12px;">You can upgrade anytime from your billing settings to raise this limit immediately.</p>
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
