import nox
from laminci.nox import build_docs, login_testuser1, run, run_pre_commit, run_pytest

# we'd like to aggregate coverage information across sessions
# and for this the code needs to be located in the same
# directory in every github action runner
# this also allows to break out an installation section
nox.options.default_venv_backend = "none"


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
def test(session):
    run(session, "uv pip install --system -e .[dev]")
    login_testuser1(session)  # TODO: remove tomorrow
    run_pytest(session)


@nox.session()
def docs(session):
    run(session, "lamin init")
    build_docs(session, strict=False)
