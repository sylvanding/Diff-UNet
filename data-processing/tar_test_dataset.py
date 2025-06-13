import argparse
import os
import sys
import tarfile


def find_test_files(data_dir):
    """
    根据 brats_data_utils_multi_label.py 中的逻辑查找测试文件。
    """
    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录不存在 -> {data_dir}", file=sys.stderr)
        sys.exit(1)

    all_dirs = os.listdir(data_dir)
    all_paths = [
        os.path.join(data_dir, d)
        for d in all_dirs
        if d.startswith("BraTS2021") and os.path.isdir(os.path.join(data_dir, d))
    ]
    all_paths.sort()

    size = len(all_paths)
    if size == 0:
        print(f"在 {data_dir} 中没有找到以 'BraTS2021' 开头的数据目录。")
        return []

    print(f"BraTS2021 数据总大小为 {size}。")
    train_size = int(0.7 * size)
    val_size = int(0.1 * size)

    # 确定测试文件
    test_files = all_paths[train_size + val_size : size]

    print(f"划分: 训练集={train_size}, 验证集={val_size}, 测试集={len(test_files)}")
    return test_files


def create_tar_archive(file_list, output_filename):
    """
    从文件/目录列表中创建 tar 归档文件。
    """
    if not file_list:
        print("没有文件需要归档。")
        return

    print(f"正在创建 tar 归档: {output_filename}")
    try:
        with tarfile.open(output_filename, "w") as tar:
            for item_path in file_list:
                # 使用 basename 作为归档内的名称，避免存储完整路径
                arcname = os.path.basename(item_path)
                print(f"添加 {item_path} 为 {arcname}")
                tar.add(item_path, arcname=arcname)
        print("归档创建成功。")
    except Exception as e:
        print(f"创建归档时出错: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 BraTS 测试数据集归档为 .tar 文件。"
    )
    parser.add_argument(
        "data_dir", type=str, help="包含 BraTS2021... 子目录的数据目录路径。"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="test_dataset.tar",
        help="输出的 .tar 文件路径。 (默认: test_dataset.tar)",
    )

    args = parser.parse_args()

    test_files = find_test_files(args.data_dir)
    create_tar_archive(test_files, args.output)
