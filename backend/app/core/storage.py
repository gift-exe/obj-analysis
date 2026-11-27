"""Object storage: local disk (served at /media) or S3/MinIO (pre-signed URLs)."""

import shutil
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings


class Storage(Protocol):
    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None: ...
    def get_file(self, key: str, dest: Path) -> Path: ...
    def local_path(self, key: str) -> Path | None: ...
    def url(self, key: str) -> str: ...
    def upload_target(self, key: str, content_type: str, video_id: str) -> tuple[str, dict[str, str]]: ...
    def delete_prefix(self, prefix: str) -> None: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError("invalid storage key")
        return path

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != dest:
            shutil.copyfile(path, dest)

    def write_stream(self, key: str, chunks) -> int:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with open(dest, "wb") as f:
            for chunk in chunks:
                f.write(chunk)
                size += len(chunk)
        return size

    def get_file(self, key: str, dest: Path) -> Path:
        return self._path(key)  # already local; no copy needed

    def local_path(self, key: str) -> Path | None:
        path = self._path(key)
        return path if path.exists() else None

    def url(self, key: str) -> str:
        return f"/media/{key}"

    def upload_target(self, key: str, content_type: str, video_id: str) -> tuple[str, dict[str, str]]:
        return f"/api/videos/{video_id}/upload", {"Content-Type": content_type}

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix), ignore_errors=True)


class S3Storage:
    def __init__(self):
        import boto3

        s = get_settings()
        self.bucket = s.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            region_name=s.s3_region,
            aws_access_key_id=s.s3_access_key,
            aws_secret_access_key=s.s3_secret_key,
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else None
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra)

    def get_file(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            self.client.download_file(self.bucket, key, str(dest))
        return dest

    def local_path(self, key: str) -> Path | None:
        return None

    def url(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=6 * 3600
        )

    def upload_target(self, key: str, content_type: str, video_id: str) -> tuple[str, dict[str, str]]:
        url = self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=3600,
        )
        return url, {"Content-Type": content_type}

    def delete_prefix(self, prefix: str) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        s = get_settings()
        _storage = S3Storage() if s.storage_backend == "s3" else LocalStorage(s.local_storage_dir)
    return _storage


def url_or_none(key: str | None) -> str | None:
    return get_storage().url(key) if key else None


def video_prefix(video_id) -> str:
    return f"videos/{video_id}"
