import tempfile

import nox

locations = "tests", "noxfile.py", "src"
nox.options.sessions = "lint", "mypy", "tests"


@nox.session(python="3.10")
def black(session):
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run(trajectory_queue)


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(trajectory_queue)
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


# flake8-black causes a warning if black would make changes
@nox.session(python=["3.10"])
def lint(session):
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
    )
    session.run(trajectory_queue)


@nox.session(python=["3.10"])
def mypy(session):
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run(trajectory_queue)


@nox.session(python=["3.10"])
def tests(session) -> None:
    args = session.posargs
    session.run(trajectory_queue)
    session.run(trajectory_queue)
