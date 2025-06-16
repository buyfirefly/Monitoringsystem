#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MELD 数据清洗脚本
-------------------------------------------------
功能:
  1. 扫描指定目录下所有 .mp4
  2. 使用 ffprobe 检测损坏或无音频的视频
  3. 将坏文件移动到 backup 目录
  4. 生成 broken.lst
  5. 自动更新同目录下的所有 CSV (删除坏样本行)

使用示例:
  python clean_meld.py --root ./MELD.Raw --check_audio --num_workers 8
"""

import argparse, subprocess, pathlib, shutil, multiprocessing as mp, sys
import pandas as pd


# ---------- ffprobe 检测 ----------------------------------------------------
def _probe_ok(filepath: pathlib.Path, check_audio=False) -> bool:
    """
    返回 True 代表文件正常; False 代表损坏/无法解析/音频为空
    """
    # 1. 检查视频流
    cmd_v = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filepath)
    ]
    if subprocess.call(cmd_v, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL) != 0:
        return False

    # 2. 可选: 检查音频流是否存在且时长>0
    if check_audio:
        try:
            dur = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(filepath)
            ], text=True).strip()
            if not dur or float(dur) == 0.0:
                return False
        except subprocess.CalledProcessError:
            return False
    return True


def _checker(arg):
    """多进程 worker 包装"""
    path, check_audio = arg
    return None if _probe_ok(path, check_audio) else path


# ---------- 主流程 ----------------------------------------------------------
def clean_dataset(root: pathlib.Path,
                  backup: pathlib.Path,
                  num_workers: int = 4,
                  check_audio: bool = False):
    mp4_files = list(root.rglob("*.mp4"))
    print(f"发现视频文件 {len(mp4_files)} 个，开始检测 …")

    # 多进程检测
    with mp.Pool(num_workers) as pool:
        bad_files = [r for r in pool.imap_unordered(
            _checker, [(p, check_audio) for p in mp4_files]) if r]

    print(f"检测完毕，损坏/空音频文件 {len(bad_files)} 个")

    if not bad_files:
        print("数据集无损坏文件，退出")
        return

    # 备份目录
    backup.mkdir(parents=True, exist_ok=True)

    # 移动坏文件
    for f in bad_files:
        dest = backup / f.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))

    # 写入列表
    broken_lst = root / "broken.lst"
    broken_lst.write_text("\n".join(str(p) for p in bad_files), encoding="utf-8")
    print(f"坏文件路径已写入 {broken_lst}")

    # 更新 CSV
    csv_paths = list(root.rglob("*.csv"))
    print(f"开始同步 {len(csv_paths)} 个 CSV …")
    for csv in csv_paths:
        df = pd.read_csv(csv)
        before = len(df)
        # 假设 CSV 里有列名 'Video' 或 'video_path'
        col = "Video" if "Video" in df.columns else "video_path"
        df = df[~df[col].apply(lambda p: str(root / p) in bad_files)]
        df.to_csv(csv, index=False)
        print(f"  {csv.name}: {before} → {len(df)} 行")

    print("\n✅ 清洗完成！记得重新跑训练脚本验证。")


# ---------- CLI ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MELD 清洗工具")
    parser.add_argument("--root", type=str, required=True,
                        help="数据集根目录，例如 ./MELD.Raw")
    parser.add_argument("--backup", type=str, default=None,
                        help="损坏文件备份目录 (默认 <root>/../broken_backup)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="并行检测进程数")
    parser.add_argument("--check_audio", action="store_true",
                        help="同时检测音频流（时长为 0 也视为损坏）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = pathlib.Path(args.root).resolve()
    backup_dir = pathlib.Path(
        args.backup) if args.backup else root_dir.parent / "broken_backup"
    clean_dataset(root_dir, backup_dir,
                  num_workers=args.num_workers,
                  check_audio=args.check_audio)