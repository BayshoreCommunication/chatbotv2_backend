import uuid
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile

from config import settings

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/ogg"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_VIDEO_BYTES = 25 * 1024 * 1024  # 25 MB


def _s3_client():
    return boto3.client(
        "s3",
        region_name=settings.DO_SPACES_REGION,
        endpoint_url=settings.DO_SPACES_ENDPOINT,
        aws_access_key_id=settings.DO_SPACES_KEY,
        aws_secret_access_key=settings.DO_SPACES_SECRET,
    )


async def upload_image(file: UploadFile) -> str:
    """Validate and upload an image to DO Spaces. Returns CDN URL."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Invalid image type. Allowed: JPG, PNG, WebP, GIF.")

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds 5 MB limit.")

    ext = (file.filename or "image").rsplit(".", 1)[-1].lower()
    key = f"{settings.DO_FOLDER_NAME}/widget/images/{uuid.uuid4()}.{ext}"

    return _upload_bytes(data, key, file.content_type)


async def upload_video(file: UploadFile) -> str:
    """Validate and upload a video to DO Spaces. Returns CDN URL."""
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise ValueError("Invalid video type. Allowed: MP4, WebM, OGG.")

    data = await file.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise ValueError("Video exceeds 25 MB limit.")

    ext = (file.filename or "video").rsplit(".", 1)[-1].lower()
    key = f"{settings.DO_FOLDER_NAME}/widget/videos/{uuid.uuid4()}.{ext}"

    return _upload_bytes(data, key, file.content_type)


def _upload_bytes(data: bytes, key: str, content_type: str) -> str:
    try:
        client = _s3_client()
        client.put_object(
            Bucket=settings.DO_SPACES_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
            ACL="public-read",
        )
        return f"{settings.DO_SPACES_CDN_URL}/{key}"
    except ClientError as e:
        raise RuntimeError(f"DO Spaces error: {e.response['Error']['Message']}") from e
