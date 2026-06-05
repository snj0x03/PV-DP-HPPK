# Label validation pipeline.
# Checks that a Roboflow YOLO export uses correct class IDs and SVC Part Numbers.
#
# Checks:
#   1. data.yaml names  → all names must exist in Excel SVC Part Number column
#   2. data.yaml order  → names must be in the same order as Excel rows
#   3. label .txt files → all class_ids must be within 0..(nc-1)
#
# Auto-fix (prompted):
#   When all names are valid but order is wrong:
#     · Remaps class_ids in all label files to match Excel order
#     · Rewrites data.yaml names in Excel order
#   When unknown names exist: reports only — fix in Roboflow and re-export.

import os

import yaml


def _load_excel_part_numbers(excel_path: str) -> list:
    """Read SVC Part Numbers from col0, skipping the header row."""
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("[validate] openpyxl is required: pip install openpyxl")

    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active
    parts = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        val = row[0]
        if val is not None:
            parts.append(str(val).strip())
    wb.close()
    return parts


def _load_data_yaml(yolo_dir: str) -> dict:
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise SystemExit(f"[validate] data.yaml not found in: {yolo_dir}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scan_label_files(yolo_dir: str, nc: int) -> dict:
    """
    Scan all .txt files under yolo_dir/labels/ (including train/valid/test sub-splits).
    Returns:
      scanned      — total files checked
      bad_files    — {filepath: [out-of-range class_ids found]}
      class_counts — {class_id: bbox_count} for valid ids
    """
    scanned = 0
    bad_files: dict = {}
    class_counts: dict = {}

    labels_root = os.path.join(yolo_dir, "labels")
    if not os.path.isdir(labels_root):
        return {"scanned": 0, "bad_files": {}, "class_counts": {}}

    for root, _, files in os.walk(labels_root):
        for fname in sorted(files):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            scanned += 1
            bad_ids = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    try:
                        cid = int(tokens[0])
                    except ValueError:
                        continue
                    if 0 <= cid < nc:
                        class_counts[cid] = class_counts.get(cid, 0) + 1
                    else:
                        bad_ids.append(cid)
            if bad_ids:
                bad_files[fpath] = bad_ids

    return {"scanned": scanned, "bad_files": bad_files, "class_counts": class_counts}


def _build_remap(names: list, excel_parts: list) -> dict:
    """
    Build {current_id: correct_id} for names in the wrong position.
    Returns empty dict if any name is missing from excel_parts (unsafe to remap).
    Only entries where current_id != correct_id are included.
    """
    excel_index = {name: i for i, name in enumerate(excel_parts)}
    if any(n not in excel_index for n in names):
        return {}
    return {
        cid: excel_index[name]
        for cid, name in enumerate(names)
        if excel_index[name] != cid
    }


def _apply_remap_labels(yolo_dir: str, remap: dict) -> int:
    """Rewrite all label files with remapped class_ids. Returns count of modified files."""
    labels_root = os.path.join(yolo_dir, "labels")
    modified = 0
    for root, _, files in os.walk(labels_root):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            new_lines = []
            changed = False
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if not tokens:
                        new_lines.append(line)
                        continue
                    try:
                        cid = int(tokens[0])
                    except ValueError:
                        new_lines.append(line)
                        continue
                    if cid in remap:
                        tokens[0] = str(remap[cid])
                        new_lines.append(" ".join(tokens) + "\n")
                        changed = True
                    else:
                        new_lines.append(line)
            if changed:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                modified += 1
    return modified


def _rewrite_yaml_order(yolo_dir: str, names: list, excel_parts: list) -> None:
    """Rewrite data.yaml names in Excel order (only names present in the dataset)."""
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    excel_index = {name: i for i, name in enumerate(excel_parts)}
    sorted_names = sorted(names, key=lambda n: excel_index.get(n, 9999))
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["names"] = sorted_names
    data["nc"] = len(sorted_names)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def validate_pipeline(yolo_dir: str, excel_path: str) -> None:
    print("\n========== Validate ==========")

    excel_parts = _load_excel_part_numbers(excel_path)
    excel_index = {name: i for i, name in enumerate(excel_parts)}
    print(f"Excel: {len(excel_parts)} SVC Part Number(s) loaded\n")

    data  = _load_data_yaml(yolo_dir)
    names = data.get("names", [])
    nc    = int(data.get("nc", len(names)))

    # ── 1. data.yaml names check ───────────────────────────────────────────────
    print(f"--- data.yaml  (nc={nc}) ---")
    unknown  = []
    misplaced = []
    for cid, name in enumerate(names):
        if name not in excel_index:
            status = "✗  NOT in Excel"
            unknown.append(name)
        elif excel_index[name] != cid:
            status = f"✗  should be class {excel_index[name]}"
            misplaced.append(cid)
        else:
            status = "✓"
        print(f"  {cid:>3}: {name:<32} {status}")

    name_ok  = not unknown
    order_ok = name_ok and not misplaced

    if not name_ok:
        print(f"\n  [FAIL] {len(unknown)} name(s) not in Excel — fix in Roboflow and re-export")
    elif not order_ok:
        print(f"\n  [FAIL] {len(misplaced)} class(es) in wrong position")
    else:
        print(f"\n  [OK] All {nc} class(es) match Excel SVC Part Numbers and order")

    # ── 2. Label file class_id check ──────────────────────────────────────────
    result       = _scan_label_files(yolo_dir, nc)
    scanned      = result["scanned"]
    bad_files    = result["bad_files"]
    class_counts = result["class_counts"]

    print(f"\n--- Label Files  (scanned: {scanned}) ---")
    label_ok = not bad_files
    if bad_files:
        print(f"  [FAIL] {len(bad_files)} file(s) with out-of-range class_id:")
        for fpath, ids in list(bad_files.items())[:10]:
            print(f"    {os.path.relpath(fpath, yolo_dir)}: ids {sorted(set(ids))}")
        if len(bad_files) > 10:
            print(f"    ... and {len(bad_files) - 10} more")
    else:
        print(f"  [OK] All class_ids within valid range (0–{nc - 1})")

    # ── 3. Per-class bbox count ────────────────────────────────────────────────
    if class_counts:
        print(f"\n--- BBox Count per Class ---")
        for cid in range(nc):
            name   = names[cid] if cid < len(names) else f"class_{cid}"
            cnt    = class_counts.get(cid, 0)
            marker = "  ← EMPTY" if cnt == 0 else ""
            print(f"  {cid:>3}: {name:<32} {cnt:>6} bbox(es){marker}")

    # ── 4. Summary + fix ──────────────────────────────────────────────────────
    print("\n--- Summary ---")
    print(f"  {'PASS' if name_ok  else 'FAIL'}  Class names in Excel")
    print(f"  {'PASS' if order_ok else 'FAIL'}  Class order matches Excel")
    print(f"  {'PASS' if label_ok else 'FAIL'}  Label file class_ids")

    if name_ok and order_ok and label_ok:
        print("\n  → Dataset is valid.")
        print("=" * 30)
        return

    if not name_ok:
        print("\n  → Correct class names in Roboflow and re-export.")
        print("=" * 30)
        return

    # All names valid — offer remap if order is wrong
    remap = _build_remap(names, excel_parts)
    if not remap:
        print("=" * 30)
        return

    print("\n  Proposed class_id remap:")
    for old_id in sorted(remap):
        print(f"    class {old_id} ({names[old_id]:<30}) → class {remap[old_id]}")

    print()
    answer = input("  Apply fix? Rewrites label files + data.yaml [y/N]: ").strip().lower()
    if answer != "y":
        print("  Skipped.")
        print("=" * 30)
        return

    n = _apply_remap_labels(yolo_dir, remap)
    _rewrite_yaml_order(yolo_dir, names, excel_parts)
    print(f"\n  ✓ {n} label file(s) updated")
    print(f"  ✓ data.yaml rewritten in Excel order")
    print("  → Re-run validate to confirm.")
    print("=" * 30)
