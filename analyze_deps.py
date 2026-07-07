import ast
import os

# Define the project root
root = r'C:\test-sandbox\be\PreVectorDeps\PreVectorChunks'

# Skip directories
skip_dirs = {'__pycache__', '.git', 'build', 'dist', 'prevectorchunks_core.egg-info', '.pytest_cache', 'migrations', 'llm-test'}

# Map import names to pip package names
import_to_package = {
    'django': 'Django',
    'corsheaders': 'django-cors-headers',
    'fastapi': 'fastapi',
    'jwt': 'PyJWT',
    'langchain_text_splitters': 'langchain-text-splitters',
    'openai': 'openai',
    'pinecone': 'pinecone',
    'dotenv': 'python-dotenv',
    'pytesseract': 'pytesseract',
    'docx': 'python-docx',
    'PyPDF2': 'PyPDF2',
    'PIL': 'pillow',
    'datasets': 'datasets',
    'pytest': 'pytest',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchaudio': 'torchaudio',
    'sentence_transformers': 'sentence-transformers',
    'gutenberg': 'py-gutenberg',
    'pathlib': 'pathlib',
    'transformers': 'transformers',
    'imageio_ffmpeg': 'imageio-ffmpeg',
    'cv2': 'opencv-python',
    'requests': 'requests',
    'langchain': 'langchain',
    'langchain_core': 'langchain-core',
    'langchain_openai': 'langchain-openai',
    'pdf2image': 'pdf2image',
    'docx2pdf': 'docx2pdf',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'fitz': 'fitz',
}

# Collect all imports
found_imports = set()

for dirpath, dirnames, filenames in os.walk(root):
    # Skip unwanted directories
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    
    for filename in filenames:
        if not filename.endswith('.py'):
            continue
        
        filepath = os.path.join(dirpath, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=filepath)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        found_imports.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        found_imports.add(module_name)
        except Exception as e:
            pass

# Map found imports to packages
matched_packages = set()
for imp in sorted(found_imports):
    if imp in import_to_package:
        matched_packages.add(import_to_package[imp])

# Read requirements
with open(os.path.join(root, 'requirements.txt'), 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Extract package name from requirement (handle version specs)
def extract_package_name(req):
    # Remove version specifiers
    for op in ['==', '~=', '>=', '<=', '!=', '>', '<']:
        if op in req:
            return req.split(op)[0]
    return req

req_map = {}
for req in requirements:
    pkg_name = extract_package_name(req)
    req_map[pkg_name] = req

print("\nREQUIREMENTS ANALYSIS:")
print("=" * 70)

print("\nUSED DEPENDENCIES (found in imports):")
print("-" * 70)
used = []
for pkg_name in sorted(req_map.keys()):
    if pkg_name in matched_packages:
        print(f"✓ {req_map[pkg_name]}")
        used.append(req_map[pkg_name])

print("\n\nUNUSED DEPENDENCIES (NOT found in imports):")
print("-" * 70)
unused = []
for pkg_name in sorted(req_map.keys()):
    if pkg_name not in matched_packages:
        print(f"✗ {req_map[pkg_name]}")
        unused.append(req_map[pkg_name])

print("\n\nSUMMARY:")
print("-" * 70)
print(f"Total requirements: {len(requirements)}")
print(f"Used: {len(used)}")
print(f"Unused: {len(unused)}")
print(f"Can remove: {len(unused)} dependencies")

print("\n\nUNUSED PACKAGES TO REMOVE:")
print("-" * 70)
for pkg in unused:
    print(f"  - {pkg}")

# Save the minimal requirements
print("\n\nCreating requirements-min.txt...")
with open(os.path.join(root, 'requirements-min.txt'), 'w') as f:
    for dep in sorted(used):
        f.write(dep + '\n')

print(f"✓ requirements-min.txt created with {len(used)} dependencies")

