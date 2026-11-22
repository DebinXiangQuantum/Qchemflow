"""The worker's result channel survives libraries that print to file descriptor 1."""

import subprocess
import sys
import textwrap

import qcl


def test_library_output_goes_to_stderr_not_into_the_result():
    code = textwrap.dedent("""
        import os
        from qcl import _worker

        stream = _worker._result_stream()
        os.write(1, b"C-level noise\\n")
        print("Python noise", flush=True)
        stream.write('{"ok": true}')
        stream.close()
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True)
    assert done.stdout == '{"ok": true}'
    assert "C-level noise" in done.stderr and "Python noise" in done.stderr


def test_a_plan_still_returns_its_result(monkeypatch):
    monkeypatch.setenv("QCL_WARM_WORKER", "0")

    @qcl.workflow
    def energy(cpu):
        return qcl.scf.hf(qcl.Molecule("H 0 0 0; H 0 0 0.74", "sto-3g")).to(cpu).energy

    value = energy.compile(qcl.Cluster.local().cpus[0]).run()
    assert value.unit == "hartree" and -1.2 < value.value < -1.0
