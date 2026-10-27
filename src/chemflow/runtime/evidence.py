from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types.results import _CAPABILITY, Evidence, EvidenceStatus

EMPIRICAL_FIELDS = ("model_card", "validity_status", "max_tanimoto", "experimental_relation",
                    "model", "value", "unit")


def empirical_payload(estimate):
    from chemflow.ids import canonical_json

    return canonical_json({"target": estimate.target, "value": estimate.value.model_dump(),
                           "diagnostics": {key: estimate.diagnostics[key]
                                           for key in EMPIRICAL_FIELDS}})


def decode_empirical_estimate(payload, store):
    from chemflow.types import Estimate

    estimate = Estimate.model_validate({**payload, "evidence": []})
    evidence = []
    for item in payload.get("evidence", []):
        if item["status"] == "empirical_calibration":
            reference = item.get("artifact_ref")
            if store is None or store.get(reference, verify=True).tobytes().decode() != (
                    empirical_payload(estimate)):
                raise ValueError("Empirical evidence differs from its verified artifact")
            evidence.append(_issue_verified(EvidenceStatus.empirical_calibration, reference, store))
        else:
            evidence.append(Evidence.model_validate(item))
    return estimate.model_copy(update={"evidence": evidence})


def _issue_verified(status: EvidenceStatus, artifact_ref: str,
                    store: ArtifactStore) -> Evidence:
    if not store.has(artifact_ref):
        raise ValueError("Evidence artifact does not exist in the executor store")
    return Evidence._runtime(status, artifact_ref, capability=_CAPABILITY)
