import pytest


@pytest.fixture(autouse=True)
def ensure_ci_workdir(request):
    """Create CI working dir inside pyfakefs when the `fs` fixture is active.

    Use request.getfixturevalue to access `fs` only if present, avoiding
    failures when pyfakefs is not used.
    """
    if "fs" in request.fixturenames:
        try:
            fs = request.getfixturevalue("fs")
            fs.create_dir("/Users/runner/work/hiclass/hiclass")
        except Exception:
            # best-effort: continue if creation fails
            pass
    yield
