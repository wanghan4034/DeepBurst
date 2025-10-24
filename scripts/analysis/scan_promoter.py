#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_promoter.py

功能：
1) 用正则 TATA[AT]A[AT][AG] 扫描 TATA box。
   - 可选提供 TSS(1-based),并限定 TATA 位点位于 TSS 上游的 [--tata-min, --tata-max] bp 范围内。
2) 采用 Takai & Jones (2002) 阈值检测 CpG 岛：
   - 长度 >= 500 bp
   - GC 含量 >= 55%
   - CpG Obs/Exp >= 0.65
   - 滑动窗口大小/步长可调(默认 500 / 1)

仅用 Python 标准库,无需外部依赖。
"""

import argparse
import re
import sys
from textwrap import dedent

TATA_REGEX = re.compile(r"TATA[AT]A[AT][AG]", re.IGNORECASE)

def read_fasta(path):
    seqs = []
    with open(path) as f:
        header = None
        buf = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seqs.append(("".join(buf)).upper())
                    buf = []
                header = line[1:]
            else:
                buf.append(line)
        if buf:
            seqs.append(("".join(buf)).upper())
    if not seqs:
        raise ValueError("FASTA 文件为空或未能解析出序列")
    if len(seqs) > 1:
        sys.stderr.write("[警告] 检测到多个序列,只取第一个用于分析。\n")
    return seqs[0]

def get_sequence(args):
    if args.seq and args.fasta:
        raise ValueError("请只提供 --seq 或 --fasta 其中之一。")
    if not args.seq and not args.fasta:
        raise ValueError("必须提供 --seq 或 --fasta。")
    if args.seq:
        return args.seq.upper()
    else:
        return read_fasta(args.fasta)

def find_tata(seq, tss_1based=None, tata_min=25, tata_max=35):
    """
    返回列表：每个元素为字典
    {
      'motif': str,
      'start0': int, 'end0': int,   # 0-based, [start0, end0)
      'start1': int, 'end1': int,   # 1-based, inclusive
      'distance_to_tss': int or None
    }
    """
    hits = []
    tss0 = None
    if tss_1based is not None:
        if tss_1based <= 0:
            raise ValueError("TSS 必须是 1-based 正整数")
        tss0 = tss_1based - 1  # 转为 0-based

    for m in TATA_REGEX.finditer(seq):
        start0, end0 = m.start(), m.end()
        start1, end1 = start0 + 1, end0  # 1-based inclusive
        entry = {
            "motif": m.group().upper(),
            "start0": start0,
            "end0": end0,
            "start1": start1,
            "end1": end1,
            "distance_to_tss": None
        }
        if tss0 is not None:
            # TATA box 通常在 TSS 上游,因此 distance = TSS - motif_start
            dist = tss0 - start0
            entry["distance_to_tss"] = dist
            if tata_min <= dist <= tata_max:
                hits.append(entry)
        else:
            hits.append(entry)
    return hits

def calc_cpg_metrics(window_seq):
    L = len(window_seq)
    C = window_seq.count('C')
    G = window_seq.count('G')
    CpG = sum(1 for i in range(L - 1) if window_seq[i:i+2] == 'CG')
    gc = (C + G) / L * 100 if L > 0 else 0.0
    obs_exp = (CpG * L) / (C * G) if C > 0 and G > 0 else 0.0
    return gc, obs_exp, C, G, CpG

def merge_intervals(intervals):
    """ intervals: list of (start, end) 0-based half-open; 返回合并后的同格式列表 """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:  # overlap or touch
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]

def detect_cpg_islands(seq, window, step, min_len, gc_thr, obs_exp_thr):
    """
    返回列表：每个元素为字典
    {
      'start0': int, 'end0': int,
      'start1': int, 'end1': int,
      'len': int,
      'GC': float,
      'ObsExp': float,
      'C': int, 'G': int, 'CpG': int
    }
    """
    L = len(seq)
    raw_hits = []
    for i in range(0, L - window + 1, step):
        w = seq[i:i+window]
        gc, obs_exp, C, G, CpG = calc_cpg_metrics(w)
        if window >= min_len and gc >= gc_thr*100 and obs_exp >= obs_exp_thr:
            raw_hits.append((i, i + window))

    merged = merge_intervals(raw_hits)

    islands = []
    for s, e in merged:
        sub = seq[s:e]
        gc, obs_exp, C, G, CpG = calc_cpg_metrics(sub)
        if (e - s) >= min_len and gc >= gc_thr*100 and obs_exp >= obs_exp_thr:
            islands.append({
                "start0": s,
                "end0": e,
                "start1": s + 1,
                "end1": e,
                "len": e - s,
                "GC": gc,
                "ObsExp": obs_exp,
                "C": C, "G": G, "CpG": CpG
            })
    return islands

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="判定特定 DNA 序列中是否包含 TATA box 与 CpG 岛(Takai & Jones 阈值默认)。",
        epilog=dedent("""\
        例子：
          1) 直接给序列：
             python scan_promoter.py --seq ACCTATATAAAGCGCG... --tss 1000 --tata-min 25 --tata-max 35
          2) 从 FASTA 读取：
             python scan_promoter.py --fasta input.fa
          3) 指定 CpG 岛扫描参数(窗口/步长/阈值)：
             python scan_promoter.py --seq ... --cpg-window 500 --cpg-step 1 --cpg-min-len 500 --cpg-gc 0.55 --cpg-obs-exp 0.65
        """)
    )
    grp_seq = parser.add_mutually_exclusive_group(required=True)
    grp_seq.add_argument("--seq", type=str, help="直接提供 DNA 序列字符串(A/C/G/T/N)")
    grp_seq.add_argument("--fasta", type=str, help="从 FASTA 文件读取序列(若多条,仅取第一条)")

    # TATA 参数
    parser.add_argument("--tss", type=int, default=None,
                        help="TSS 的 1-based 位置；若提供则只保留位于 [--tata-min, --tata-max] 上游距离内的 TATA 命中")
    parser.add_argument("--tata-min", type=int, default=25, help="TATA 到 TSS 的最小上游距离(bp,默认 25)")
    parser.add_argument("--tata-max", type=int, default=35, help="TATA 到 TSS 的最小上游距离(bp,默认 35)")

    # CpG 参数(Takai & Jones 默认)
    parser.add_argument("--cpg-window", type=int, default=500, help="CpG 扫描窗口大小(默认 500)")
    parser.add_argument("--cpg-step", type=int, default=1, help="CpG 扫描步长(默认 1)")
    parser.add_argument("--cpg-min-len", type=int, default=500, help="CpG 岛最小长度(默认 500)")
    parser.add_argument("--cpg-gc", type=float, default=0.55, help="GC 含量阈值(默认 0.55 = 55%%)")
    parser.add_argument("--cpg-obs-exp", type=float, default=0.65, help="Obs/Exp 阈值(默认 0.65)")

    args = parser.parse_args()

    seq = get_sequence(args)
    if not seq:
        print("序列为空。", file=sys.stderr)
        sys.exit(1)

    # TATA
    tata_hits = find_tata(seq, tss_1based=args.tss, tata_min=args.tata_min, tata_max=args.tata_max)

    # CpG
    cpg_islands = detect_cpg_islands(
        seq,
        window=args.cpg_window,
        step=args.cpg_step,
        min_len=args.cpg_min_len,
        gc_thr=args.cpg_gc,
        obs_exp_thr=args.cpg_obs_exp
    )

    # 输出
    print("# === TATA box 命中(0-based 半开区间 & 1-based 闭区间) ===")
    if tata_hits:
        print("motif\tstart0\tend0\tstart1\tend1\tdistance_to_tss")
        for h in tata_hits:
            print("{motif}\t{start0}\t{end0}\t{start1}\t{end1}\t{dist}".format(
                motif=h["motif"],
                start0=h["start0"], end0=h["end0"],
                start1=h["start1"], end1=h["end1"],
                dist=h["distance_to_tss"] if h["distance_to_tss"] is not None else "NA"
            ))
    else:
        print("未检测到 TATA box(或未满足 TSS 上游窗口过滤条件)。")

    print("\n# === CpG 岛(Takai & Jones 默认阈值,可通过参数修改) ===")
    if cpg_islands:
        print("start0\tend0\tstart1\tend1\tlen\tGC%\tObs/Exp\t#C\t#G\t#CpG")
        for isl in cpg_islands:
            print("{s0}\t{e0}\t{s1}\t{e1}\t{l}\t{gc:.2f}\t{oe:.3f}\t{C}\t{G}\t{CpG}".format(
                s0=isl["start0"], e0=isl["end0"], s1=isl["start1"], e1=isl["end1"],
                l=isl["len"], gc=isl["GC"], oe=isl["ObsExp"], C=isl["C"], G=isl["G"], CpG=isl["CpG"]
            ))
    else:
        print("未检测到符合阈值的 CpG 岛。")

if __name__ == "__main__":
    main()