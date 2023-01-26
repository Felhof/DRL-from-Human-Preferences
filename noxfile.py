import tempfile

import nox

locations = "tests", "noxfile.py", "src"
nox.options.sessions = "lint", "mypy", "tests"


@nox.session(python="3.10")
def black(session):
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as tmp:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={tmp.name}",
            external=True,
        )
        # remove lines with extra constrains e.g. gymnasium[accept-rom-license]
        # as they cause a pip error
        requirements = tmp.readlines()
        tmp.seek(0)
        for line in requirements:
            if "[" not in str(line):
                tmp.write(line)
        tmp.truncate()
        session.install(f"--constraint={tmp.name}", *args, **kwargs)


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
    session.run("flake8", *args)


@nox.session(python=["3.10"])
def mypy(session):
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python=["3.10"])
def tests(session) -> None:
    args = session.posargs
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
