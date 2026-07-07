# How to Use pyproject.toml for Dependency Management

Your project is now using **pyproject.toml** as the primary dependency source. Here's how to work with it:

## 1. **Install dependencies from pyproject.toml**

```bash
# Install in development mode (editable install)
# This installs the core library + all dependencies
pip install -e core/

# Or, if you want just the dependencies without the package:
pip install -r requirements.txt
```

## 2. **Directory Structure**

```
PreVectorChunks/
├── core/                           # Core library package
│   ├── pyproject.toml             # ← PRIMARY dependency source
│   ├── prevectorchunks_core/
│   └── ...
├── requirements.txt               # ← Synced from core/pyproject.toml
└── (main Django/FastAPI app)
```

## 3. **Workflow**

### Add a new dependency:
```bash
# Edit core/pyproject.toml, add to the `dependencies` list
# Example:
#   "new-package~=1.2.3",

# Then sync to requirements.txt (or just use pyproject.toml directly)
```

### Update dependencies:
```bash
# Edit core/pyproject.toml with new versions
# Then update requirements.txt to match
```

### Install/update everything:
```bash
pip install -e core/
# or
pip install -r requirements.txt
```

## 4. **Why pyproject.toml is Better**

| Feature                | pyproject.toml | requirements.txt |
|------------------------|---|---|
| Package metadata       | ✓ Includes version, author, description | ✗ Just dependencies |
| Build system           | ✓ Defined (setuptools) | ✗ N/A |
| Releasability          | ✓ Package can be published to PyPI | ✗ Not a package |
| Single source of truth | ✓ One place to manage versions | ✗ Manual sync needed |
| Developmentp mode      | ✓ pip install -e ./ works | ✗ Need requirements.txt |

## 5. **Current Setup**

✓ **core/pyproject.toml** - Contains all production dependencies (30 packages)
✓ **requirements.txt** - Exact mirror for easy pip install

Note: `pytest` is included in requirements.txt but NOT in pyproject.toml (it's test-only)

## 6. **Recommended Practice**

For a Django/FastAPI project with a separate core library:

```
core/pyproject.toml          → Production dependencies for the core library
requirements.txt             → All deps (core + django + optional test deps)
requirements-dev.txt         → Just test/lint tools (pytest, black, flake8, etc.)
```

If you want to separate dev dependencies:
```bash
pip install -e core/              # Just production deps
pip install -r requirements-dev.txt # Add test tools
```

## 7. **Current Status**

✓ pyproject.toml cleaned up (removed unused deps)
✓ requirements.txt synced with pyproject.toml
✓ Removed unused dependency: py-gutenberg
✓ Version conflicts resolved (torch 2.6.0, numpy 2.2.6, etc.)

Your project is now using **pyproject.toml** as the primary dependency source!

