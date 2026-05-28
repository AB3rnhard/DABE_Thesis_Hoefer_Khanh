import importlib.util
import subprocess
import sys


def _ensure(pip_name: str, import_name=None) -> None:
    """Install a package quietly when its import is unavailable."""
    check = import_name or pip_name
    if importlib.util.find_spec(check) is None:
        print(f'  Installing {pip_name} ...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name, '-q'])