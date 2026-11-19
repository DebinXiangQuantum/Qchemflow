import pytest
from test_m3a_task import spec

from chemflow.diagnostics import CheckError
from chemflow.signing import TaskSigner, TaskVerifier, generate_keypair


def test_encrypted_ed25519_roundtrip_and_tampering():
    private, public = generate_keypair(b"human passphrase")
    assert b"ENCRYPTED PRIVATE KEY" in private
    with pytest.raises((TypeError, ValueError)):
        TaskSigner(private, b"wrong")
    signer = TaskSigner(private, b"human passphrase")
    verifier = TaskVerifier(public)
    assert signer.fingerprint == verifier.fingerprint
    record = signer.freeze(spec(), "human")
    verifier.verify(record)
    newer = signer.freeze(spec(), "human", predecessor=record)
    assert newer.version == 2 and newer.predecessor_hash == record.hash
    verifier.verify(newer)
    with pytest.raises(CheckError):
        verifier.verify(record.model_copy(update={"approver": "attacker"}))
    with pytest.raises(CheckError):
        verifier.verify_bytes(b"different", signer.sign(b"payload"))


def test_other_key_cannot_sign():
    private, public = generate_keypair(b"human")
    attacker, _ = generate_keypair(b"attacker")
    verifier = TaskVerifier(public)
    verifier.verify(TaskSigner(private, b"human").freeze(spec(), "human"))
    with pytest.raises(CheckError):
        verifier.verify(TaskSigner(attacker, b"attacker").freeze(spec(), "human"))
