#!/bin/bash

# 定义所有要测试的数据集
test_files=(
    "ArchiveII"
    "TestSetB"
    "bprna_1m"
    "bprna_new"
    "16s"
    "5s"
    "grp1"
    "grp2"
    "RNaseP"
    "srp"
    "telomerase"
    "tmRNA"
    "tRNA"
    "short"
    "medium"
    "pes"
    "pes_bprnanew"
    "pes_bprna1m"
    "bprna_pes"
    "rivas_test"
)

# 日志目录
log_dir="/home/chenjingjing/Models/UFold/UFold/data/test_new"

# 确保日志目录存在
mkdir -p "$log_dir"

echo "开始逐个测试所有数据集..."
echo "总共需要测试: ${#test_files[@]} 个数据集"

# 逐个执行测试
for test_file in "${test_files[@]}"; do
    log_file="${log_dir}/${test_file}.log"

    echo "==========================================="
    echo "开始测试: $test_file"
    echo "日志文件: $log_file"

    # 执行测试命令 - 每个数据集单独运行
    nohup python ufold_test.py --test_files "$test_file" > "$log_file" 2>&1 &

    # 获取进程ID
    pid=$!
    echo "测试进程ID: $pid"

    # 等待当前测试完成再开始下一个
    echo "等待测试完成..."
    wait $pid

    # 检查测试是否成功完成
    if [ $? -eq 0 ]; then
        echo "✓ $test_file 测试完成"
    else
        echo "✗ $test_file 测试失败"
    fi

    echo "-------------------------------------------"
done

echo "==========================================="
echo "所有测试完成！"
echo "日志文件保存在: $log_dir"