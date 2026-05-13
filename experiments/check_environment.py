import argparse
import importlib
import platform
import sys

from packaging.version import InvalidVersion, Version

try:
    from importlib import metadata
except Exception:  # pragma: no cover
    try:
        import importlib_metadata as metadata
    except Exception:  # pragma: no cover
        metadata = None


EXPECTED = {
    "numpy": "1.21.6",
    "pandas": "1.1.5",
    "scipy": "1.7.3",
    "scikit-learn": "1.0.2",
    "PyYAML": "6.0.1",
    "packaging": "24.0",
    "matplotlib": "2.2.3",
    "seaborn": "0.9.0",
    "torch": "1.13.1",
    "lightgbm": "3.3.5",
}


IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
}


def installed_version(name):
    module_name = IMPORT_NAMES.get(name, name)
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return distribution_version(name)
    return getattr(module, "__version__", None) or distribution_version(name)


def distribution_version(name):
    if metadata is None:
        return None
    try:
        return metadata.version(name)
    except Exception:
        return None


def same_version(current, expected):
    if current is None:
        return False
    try:
        return Version(str(current)).base_version == Version(str(expected)).base_version
    except InvalidVersion:
        return str(current) == str(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit with a non-zero code if versions differ")
    args = parser.parse_args()

    py_expected = (3, 7)
    py_current = sys.version_info[:2]
    mismatches = []
    print("Python:", platform.python_version())
    if py_current != py_expected:
        mismatches.append(("python", "%d.%d" % py_expected, "%d.%d" % py_current))

    for name, expected in EXPECTED.items():
        current = installed_version(name)
        print("%-14s expected=%-10s current=%s" % (name, expected, current or "missing"))
        if not same_version(current, expected):
            mismatches.append((name, expected, current or "missing"))

    if mismatches:
        print("\nEnvironment differs from the report reproduction environment:")
        for name, expected, current in mismatches:
            print("  %s: expected %s, current %s" % (name, expected, current))
        print("\nThe scripts may still run, but exact inventory-cost values are not guaranteed.")
        if args.strict:
            return 1
    else:
        print("\nEnvironment matches the report reproduction environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
