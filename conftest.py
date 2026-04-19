def pytest_sessionstart(session):
    """Ensure CI working directory exists in pyfakefs to avoid chdir errors.

    Pytest-cov may call ``os.chdir`` to the repository topdir before tests
    run. When pyfakefs patches ``os``, that chdir fails if the path isn't
    present in the fake filesystem. This hook creates the expected path
    inside the fake filesystem (if the pyfakefs plugin is active).
    """
    plugin = session.config.pluginmanager.getplugin("pyfakefs")
    if plugin is None:
        return
    fs = getattr(plugin, "fs", None)
    if fs is None:
        return
    try:
        fs.create_dir("/Users/runner/work/hiclass/hiclass")
    except Exception:
        # best-effort: if creation fails, continue and let pytest report errors
        pass
