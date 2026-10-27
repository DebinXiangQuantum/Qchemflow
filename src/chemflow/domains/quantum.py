"""Opt-in sampled quantum algorithms, preserving the verification-only core."""

from chemflow.quantum.vqe import register_sampled_vqe


def register(registry):
    register_sampled_vqe(registry)
