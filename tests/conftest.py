import types
import sys


def _make_module(name):
    m = types.ModuleType(name)
    m.__file__ = f"<fake {name}>"
    return m


def pytest_configure():
    """No-op: use the real `amigo` package installed in the environment.

    Previously we injected lightweight stubs for `amigo` so imports would
    succeed in environments without the real package. The real `amigo`
    dependency is now installed in CI and local dev environments, so tests
    should exercise the actual package. Keep this hook as a no-op to avoid
    interfering with the real installation.
    """
    return
