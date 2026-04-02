import os
import shutil

# ===== CONFIG =====
OLD_BASE = ""  # optional: keep empty if paths in txt are absolute
NEW_DIR = "data/good_jsons"
TXT_FILE = "scripts/good_jsons.txt"
NEW_TXT_FILE = "scripts/good_jsons_clean.txt"  # safer (don’t overwrite initially)

# ===== SETUP =====
os.makedirs(NEW_DIR, exist_ok=True)

copied = []
missing = []

# ===== PROCESS =====
with open(TXT_FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for path in lines:
    # If paths are relative, prepend OLD_BASE
    full_path = os.path.join(OLD_BASE, path) if OLD_BASE else path

    filename = os.path.basename(path)
    new_path = os.path.join(NEW_DIR, filename)

    if os.path.exists(full_path):
        shutil.copy2(full_path, new_path)
        copied.append(filename)
    else:
        print(f"❌ Missing: {full_path}")
        missing.append(full_path)

# ===== WRITE CLEAN TXT =====
# with open(NEW_TXT_FILE, "w") as f:
#     for fname in copied:
#         f.write(fname + "\n")

# ===== SUMMARY =====
print("\n✅ Done")
print(f"Copied: {len(copied)} files")
print(f"Missing: {len(missing)} files")

if missing:
    print("\nMissing files:")
    for m in missing:
        print(m)