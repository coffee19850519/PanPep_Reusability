import argparse
from pathlib import Path
from datetime import datetime


def norm(s: str) -> str:
    # 忽略首尾空格 + 忽略大小写
    return s.strip().casefold()


def main():
    default_input = "/fs/ess/PAS1475/Fei/code/PanPep_train/Control_dataset.txt"
    default_ref = "/fs/ess/PAS1475/Fei/code/PanPep_train/meta.txt"
    default_output = f"/fs/ess/PAS1475/Fei/code/Control_dataset_clean_{datetime.now():%Y%m%d_%H%M%S}.txt"

    parser = argparse.ArgumentParser(
        description="从输入文件中删除在参考文件中出现的行（忽略大小写和首尾空格），输出到新文件。"
    )
    parser.add_argument("--input", default=default_input, help="输入文件路径")
    parser.add_argument("--ref", default=default_ref, help="参考文件路径")
    parser.add_argument("--output", default=default_output, help="输出文件路径（新文件）")
    args = parser.parse_args()

    input_path = Path(args.input)
    ref_path = Path(args.ref)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    if not ref_path.is_file():
        raise FileNotFoundError(f"找不到参考文件: {ref_path}")
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，请换个名字: {output_path}")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("输出文件不能和输入文件同名（避免覆盖）。")

    # 先加载参考文件（meta.txt）
    ref_keys = set()
    with ref_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            ref_keys.add(norm(line))

    total = 0
    removed = 0
    kept = 0

    # 流式处理大文件
    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as fin, \
         output_path.open("w", encoding="utf-8", newline="") as fout:
        for line in fin:
            total += 1
            if norm(line) in ref_keys:
                removed += 1
                continue
            fout.write(line)
            kept += 1

    print("完成")
    print(f"输入文件: {input_path}")
    print(f"参考文件: {ref_path}")
    print(f"输出文件: {output_path}")
    print(f"原始行数: {total}")
    print(f"输出行数: {kept}")
    print(f"删除行数: {removed}")


if __name__ == "__main__":
    main()