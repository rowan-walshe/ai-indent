import subprocess

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'

git_repos = [
    "https://github.com/AdaCore/Ada_Drivers_Library.git",
    "https://github.com/AdaCore/gnatstudio.git",
    "https://github.com/AdaCore/spark2014.git",
    "https://github.com/AdaCore/ada_language_server.git",
    "https://github.com/AdaCore/gnat-llvm.git",
    "https://github.com/AdaCore/libadalang.git",
    "https://github.com/AdaCore/aws.git",
    "https://github.com/AdaCore/RecordFlux.git",
    "https://github.com/AdaCore/learn.git",
    "https://github.com/AdaCore/gtkada.git",
    "https://github.com/AdaCore/gprbuild.git",
    "https://github.com/AdaCore/bb-runtimes.git",
    "https://github.com/AdaCore/svd2ada.git",
    "https://github.com/AdaCore/VSS.git",
    "https://github.com/AdaCore/gnatcoll-core.git",
    "https://github.com/AdaCore/Certyflie.git",
    "https://github.com/AdaCore/gnatcoverage.git",
]

if __name__ == "__main__":
    for repo in git_repos:
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo], cwd=RAW_DATA_DIR)
        except Exception as e:
            print(f"Failed to clone '{repo}'. {e}")