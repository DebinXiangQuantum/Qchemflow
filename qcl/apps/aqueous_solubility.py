"""Offline ESOL and GSE estimates with applicability evidence."""

import qcl


@qcl.workflow
def aqueous_solubility(smiles, cpu, melting_point_C=None):
    if melting_point_C is None:
        return qcl.solubility.esol(smiles).to(cpu)
    return qcl.solubility.gse(smiles, melting_point_C).to(cpu)


if __name__ == "__main__":
    plan = aqueous_solubility.compile("CCO", qcl.Cluster.local().cpus[0])
    print(plan.explain())
    print(plan.run())
