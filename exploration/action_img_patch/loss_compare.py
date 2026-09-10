import re
import glob
import statistics as st


def parse(files):
    data = {}
    for fn in files:
        txt = open(fn, errors="ignore").read()
        txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
        txt = re.sub(r"\s+", " ", txt)
        # step=N/41940 and loss=... are separated by e.g. "trainer.py:1170"
        for seg in re.split(r"step=", txt)[1:]:
            m_s = re.match(r"(\d+)/41940", seg)
            if not m_s:
                continue
            head = seg[:300]
            m_l = re.search(r"loss=([\d.]+) loss_action=([\d.]+) loss_video=([\d.]+)", head)
            if not m_l:
                continue
            s = int(m_s.group(1))
            vals = (float(m_l.group(1)), float(m_l.group(2)), float(m_l.group(3)))
            data.setdefault(s // 1000, []).append(vals)
    return data


def avg(d, k, i):
    v = d.get(k)
    return sum(x[i] for x in v) / len(v) if v else None


base = parse(sorted(glob.glob("slurm_logs/ap_base_2node_854*.out") + glob.glob("slurm_logs/ap_base_2node_856*.out")))
patch = parse(sorted(glob.glob("slurm_logs/ap_patch_2node_854*.out") + glob.glob("slurm_logs/ap_patch_2node_856*.out")))

hdr = ("bucket", "video.base", "video.patch", "dVideo(p-b)", "act.base", "act.patch")
print("%8s | %10s %11s %11s | %8s %9s" % hdr)
common = []
for k in sorted(set(base) | set(patch)):
    vb, vp = avg(base, k, 2), avg(patch, k, 2)
    ab, ap = avg(base, k, 1), avg(patch, k, 1)
    dv = ("%+.4f" % (vp - vb)) if (vb is not None and vp is not None) else "-"
    if vb is not None and vp is not None:
        common.append(vp - vb)
    fmt = lambda x: ("%.4f" % x) if x is not None else "-"
    print("%7dk | %10s %11s %11s | %8s %9s" % (k, fmt(vb), fmt(vp), dv, fmt(ab), fmt(ap)))

if len(common) > 1:
    print()
    print("overlap dVideo(patch-base): mean=%+.4f  std=%.4f  n=%d buckets"
          % (st.mean(common), st.stdev(common), len(common)))
