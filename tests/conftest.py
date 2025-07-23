import shutil

import lamindb_setup as ln_setup


def pytest_sessionstart():
    ln_setup.init(storage="./testdb")


def pytest_sessionfinish():
    shutil.rmtree("./testdb")
    ln_setup.delete("testdb", force=True)
