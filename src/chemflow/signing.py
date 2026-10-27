"""Human-held Ed25519 task signing and public-key verification."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from chemflow.agent.task import TaskRecord, TaskSpec
from chemflow.diagnostics import AdapterError
from chemflow.ids import canonical_json, content_hash


def generate_keypair(passphrase: bytes) -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    return (private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.BestAvailableEncryption(passphrase)),
            private.public_key().public_bytes(serialization.Encoding.PEM,
                                              serialization.PublicFormat.SubjectPublicKeyInfo))


def _fingerprint(key: Ed25519PublicKey) -> str:
    return hashlib.sha256(key.public_bytes(serialization.Encoding.Raw,
                                           serialization.PublicFormat.Raw)).hexdigest()


class TaskSigner:
    def __init__(self, private_pem: bytes, passphrase: bytes):
        key = serialization.load_pem_private_key(private_pem, password=passphrase)
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("Expected Ed25519 private key")
        self._key = key
        self.fingerprint = _fingerprint(key.public_key())

    def sign(self, payload_bytes: bytes) -> bytes:
        return self._key.sign(payload_bytes)

    def freeze(self, spec: TaskSpec, approver: str,
               predecessor: TaskRecord | None = None) -> TaskRecord:
        if not approver:
            raise ValueError("Approver identity is required")
        if predecessor is not None:
            TaskVerifier(self._key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
            ).verify(predecessor)
        record = TaskRecord(spec=spec, content_hash=content_hash(spec), approver=approver,
                            timestamp=datetime.now(UTC), version=predecessor.version + 1
                            if predecessor else 1,
                            predecessor_hash=predecessor.hash if predecessor else None,
                            key_fingerprint=self.fingerprint, signature="")
        payload = canonical_json(record.model_dump(mode="json", exclude={"signature"})).encode()
        return record.model_copy(update={"signature": self.sign(payload).hex()})


class TaskVerifier:
    def __init__(self, public_pem: bytes):
        key = serialization.load_pem_public_key(public_pem)
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("Expected Ed25519 public key")
        self._key = key
        self.fingerprint = _fingerprint(key)

    def verify_bytes(self, payload: bytes, signature: bytes) -> None:
        try:
            self._key.verify(signature, payload)
        except (InvalidSignature, ValueError, TypeError):
            raise AdapterError("CF-E120", None, "Signature verification failed",
                               "Obtain a new human signature") from None

    def verify(self, record: TaskRecord) -> None:
        try:
            if record.key_fingerprint != self.fingerprint or record.content_hash != content_hash(
                    record.spec):
                raise ValueError
            payload = canonical_json(record.model_dump(mode="json", exclude={"signature"}))
            self.verify_bytes(payload.encode(), bytes.fromhex(record.signature))
        except (ValueError, TypeError, AttributeError):
            raise AdapterError("CF-E120", None, "Task record integrity failure",
                               "Obtain a new approved task record") from None
