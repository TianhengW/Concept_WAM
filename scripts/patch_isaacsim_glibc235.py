#!/usr/bin/env python3
"""Retarget GLIBC_2.35 version requirements in ELF shared objects to an older
version node that the host glibc (2.34 on Rocky 9) actually provides.

Isaac Sim 5.1 wheels are built on Ubuntu 22.04 (glibc 2.35).  The only symbol
that ends up bound to GLIBC_2.35 in the shipped binaries is `hypot`, whose
version node was bumped in glibc 2.35 without changing its ABI; older glibc
still exports `hypot@GLIBC_2.2.5`.  Rewriting the Vernaux entry (hash + name
offset) makes the dynamic loader accept the library on glibc 2.34.

Usage:
  patch_glibc_verneed.py [--check] FILE_OR_DIR ...

Without --check the files are patched in place (a .orig-glibc235 backup is
kept next to each patched file).  Exit status 0 = nothing left that requires
GLIBC_2.35, 1 = at least one file still requires it (unfixable symbol).
"""
import os
import struct
import sys

from elftools.elf.elffile import ELFFile

BAD_VERSION = "GLIBC_2.35"
# Symbols known to exist under an older version node in glibc <= 2.34.
FALLBACK = {"hypot": "GLIBC_2.2.5", "hypotf": "GLIBC_2.2.5", "hypotl": "GLIBC_2.2.5"}


def elf_hash(name: bytes) -> int:
    h = 0
    for c in name:
        h = (h << 4) + c
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def analyse(path):
    """Return (needs_bad, symbols_bound_to_bad, plan) for one ELF file."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    if data[:4] != b"\x7fELF":
        return None
    elf = ELFFile(open(path, "rb"))
    verneed = elf.get_section_by_name(".gnu.version_r")
    versym = elf.get_section_by_name(".gnu.version")
    dynsym = elf.get_section_by_name(".dynsym")
    dynstr = elf.get_section_by_name(".dynstr")
    if verneed is None or versym is None or dynsym is None:
        return None
    strtab = dynstr.data()
    sec_off = verneed["sh_offset"]
    bad_aux = []  # (aux_file_offset, vna_other, lib_name)
    off = 0
    while True:
        vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, sec_off + off)
        lib = strtab[vn_file:strtab.index(b"\0", vn_file)].decode()
        aoff = off + vn_aux
        for _ in range(vn_cnt):
            vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, sec_off + aoff)
            name = strtab[vna_name:strtab.index(b"\0", vna_name)].decode()
            if name == BAD_VERSION:
                bad_aux.append((sec_off + aoff, vna_other, lib))
            if vna_next == 0:
                break
            aoff += vna_next
        if vn_next == 0:
            break
        off += vn_next
    if not bad_aux:
        return (False, [], [])
    bad_idx = {other: lib for _, other, lib in bad_aux}
    bound = []
    for sym, ver in zip(dynsym.iter_symbols(), versym.iter_symbols()):
        n = ver["ndx"]
        if isinstance(n, int) and (n & 0x7FFF) in bad_idx:
            bound.append(sym.name)
    plan = []
    for aux_off, other, lib in bad_aux:
        targets = {FALLBACK.get(s) for s in bound}
        if None in targets or len(targets) != 1:
            plan.append((aux_off, other, lib, None))
            continue
        target = targets.pop()
        tpos = strtab.find(target.encode() + b"\0")
        if tpos < 0 or strtab[tpos - 1:tpos] not in (b"\0", b""):
            plan.append((aux_off, other, lib, None))
            continue
        plan.append((aux_off, other, lib, (target, tpos)))
    return (True, bound, plan)


def patch(path, plan):
    with open(path, "rb") as f:
        data = bytearray(f.read())
    backup = path + ".orig-glibc235"
    if not os.path.exists(backup):
        with open(backup, "wb") as f:
            f.write(data)
    for aux_off, _other, _lib, target in plan:
        name, tpos = target
        struct.pack_into("<I", data, aux_off, elf_hash(name.encode()))
        struct.pack_into("<I", data, aux_off + 8, tpos)
    with open(path, "wb") as f:
        f.write(data)


def iter_files(args):
    for a in args:
        if os.path.isdir(a):
            for root, _dirs, files in os.walk(a):
                for fn in files:
                    p = os.path.join(root, fn)
                    if fn.endswith(".orig-glibc235") or os.path.islink(p) or os.path.getsize(p) < 1024:
                        continue
                    with open(p, "rb") as f:
                        if f.read(4) != b"\x7fELF":
                            continue
                    yield p
        else:
            yield a


def main():
    check = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--check"]
    unfixed = 0
    for p in iter_files(args):
        res = analyse(p)
        if res is None or not res[0]:
            continue
        _, bound, plan = res
        fixable = all(t[3] is not None for t in plan)
        print(f"{p}: requires {BAD_VERSION} for {sorted(set(bound))} -> {'fixable' if fixable else 'NOT fixable'}")
        if not fixable:
            unfixed += 1
            continue
        if check:
            continue
        patch(p, plan)
        again = analyse(p)
        if again and again[0]:
            print(f"  patch verification FAILED for {p}")
            unfixed += 1
        else:
            print(f"  patched -> {plan[0][3][0]}")
    sys.exit(1 if unfixed else 0)


if __name__ == "__main__":
    main()
