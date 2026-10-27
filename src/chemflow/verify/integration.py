"""Production-time checking and content-addressed certificate persistence."""

from pathlib import Path

from chemflow.ids import canonical_json, content_hash
from chemflow.relations import TargetRelation
from chemflow.types import EnergyCurve, Estimate, SpectralEstimate, Spectrum
from chemflow.verify.kernel import check_certificate
from chemflow.verify.prover import prove


def certify_results(program, registry, computed, directory=None):
    certificates = {}
    for node in program.nodes:
        value = computed.get(node.id)
        if node.op in ("logs_esol", "logs_gse") and isinstance(value, dict):
            certificate = prove(program, node.id, registry=registry, computed=computed)
            verdict = check_certificate(certificate, program, registry=registry, computed=computed)
            identifier = content_hash(certificate)
            computed[node.id] = {**value, "relation_status": "proved" if verdict.valid else
                                 "declared", "relation_certificate": identifier,
                                 "relation": certificate["relation"],
                                 "experimental_relation": "model_change"}
            certificates[node.id] = certificate
            if directory is not None:
                path = Path(directory) / "certificates"
                path.mkdir(parents=True, exist_ok=True)
                (path / f"{identifier}.json").write_text(canonical_json(certificate) + "\n")
            continue
        if not isinstance(value, (Estimate, SpectralEstimate, Spectrum, EnergyCurve)):
            continue
        certificate = prove(program, node.id, registry=registry, computed=computed)
        verdict = check_certificate(certificate, program, registry=registry, computed=computed)
        identifier = content_hash(certificate)
        diagnostics = {**value.diagnostics, "relation_status": "proved" if verdict.valid else "declared",
                       "relation_certificate": identifier}
        if value.diagnostics.get("relation_status") == "refuted":
            diagnostics["relation_status"] = "refuted"
        relation = value.relation
        if verdict.valid:
            relation = TargetRelation(certificate["relation"])
            if relation != value.relation:
                diagnostics.setdefault("declared_relation", value.relation.value)
        if (verdict.valid and relation == "equivalent" and
                node.op == "empirical_estimate" and computed.get("_artifacts") is not None):
            import numpy as np

            from chemflow.runtime.evidence import _issue_verified, empirical_payload
            from chemflow.types import EvidenceStatus

            store = computed["_artifacts"]
            artifact = store.put(np.frombuffer(empirical_payload(value).encode(), dtype=np.uint8))
            status = (EvidenceStatus.empirical_calibration if diagnostics["model"] == "esol"
                      and diagnostics["validity_status"] == "inside_domain" else
                      EvidenceStatus.diagnostic_only)
            value = value.model_copy(update={"evidence": [_issue_verified(status, artifact, store)]})
        if not verdict.valid:
            diagnostics["relation_proof_reasons"] = list(verdict.reasons)
        computed[node.id] = (value if relation == value.relation and diagnostics == value.diagnostics
                             else value.model_copy(update={"relation": relation,
                                                           "diagnostics": diagnostics}))
        certificates[node.id] = certificate
        if directory is not None:
            path = Path(directory) / "certificates"
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{identifier}.json").write_text(canonical_json(certificate) + "\n")
    computed["relation_certificates"] = certificates


def relation_fields(estimate):
    return {key: estimate.diagnostics.get(key, "declared" if key == "relation_status" else None)
            for key in ("relation_status", "relation_certificate")}
