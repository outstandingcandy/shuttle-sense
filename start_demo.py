#!/usr/bin/env python3
"""
演示版启动脚本 - ShuttleSense Web应用
无需加载AI模型，使用模拟数据进行演示
"""

import os
import sys

def main():
    print("🏸 启动 ShuttleSense 演示版网页应用...")
    print("=" * 50)
    
    # Check if required dependencies are installed
    try:
        import flask  # noqa: F401
        print("✓ Flask 已安装")
    except ImportError:
        print("❌ Flask 未安装，请运行: pip install flask")
        return False
    
    try:
        import yaml  # noqa: F401
        print("✓ PyYAML 已安装")
    except ImportError:
        print("❌ PyYAML 未安装，请运行: pip install pyyaml")
        return False
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        print("❌ 配置文件 config.yaml 未找到")
        return False
    else:
        print("✓ 配置文件已找到")
    
    # Check required directories
    dirs_to_check = ['uploads', 'templates', 'static']
    for directory in dirs_to_check:
        if os.path.exists(directory):
            print(f"✓ 目录 {directory} 存在")
        else:
            print(f"❌ 目录 {directory} 不存在")
            return False
    
    print("\n" + "=" * 50)
    print("所有检查通过！启动演示版网页应用...")
    print("* 演示版使用模拟数据，无需AI模型")
    print("* 上传的视频会生成示例分析结果")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("=" * 50 + "\n")
    
    # Import and run the demo app
    try:
        from app_demo import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)