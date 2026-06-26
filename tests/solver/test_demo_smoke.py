from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("z3")


def test_demo_runs_and_reports_optimum():
    out = subprocess.run(
        [sys.executable, "-m", "examples.z3_demo"],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert "optimum" in out.stdout.lower()
