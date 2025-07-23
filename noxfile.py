import os
import subprocess

import nox
from laminci.nox import build_docs, run, run_pre_commit, run_pytest

nox.options.default_venv_backend = "none"
IS_PR = os.getenv("GITHUB_EVENT_NAME") != "push"


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
def test(session):
    run(session, "uv pip install --system -e .[dev]")
    run_pytest(session)


@nox.session()
def docs(session):
    run(session, "lamin init")  # shouldn't be necessary
    build_docs(session, strict=False)
    if not IS_PR:
        subprocess.run(
            "lndocs --strip-prefix --format text --error-on-index",  # --strict back
            shell=True,  # noqa: S602
        )
