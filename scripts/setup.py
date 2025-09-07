"""
SEKI 項目設置腳本
安裝依賴並配置環境
"""

import subprocess
import sys
import os


def install_requirements():
    """安裝必要的 Python 包"""
    requirements = [
        "torch>=1.9.0",
        "numpy>=1.19.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "notebook>=6.0.0"
    ]
    
    print("Installing Python packages...")
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    return True


def create_project_structure():
    """創建項目目錄結構"""
    directories = [
        "seki",
        "experiments", 
        "scripts",
        "results",
        "tests",
        "docs/figures",
        "data"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def setup_environment():
    """設置環境變量和配置"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 添加項目根目錄到 Python 路徑
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    print(f"✓ Added {base_dir} to Python path")


def verify_installation():
    """驗證安裝是否成功"""
    try:
        import numpy
        import torch
        import sklearn
        import matplotlib
        import scipy
        print("✓ All core dependencies verified")
        
        # 嘗試導入 SEKI 模組
        try:
            import seki
            print("✓ SEKI package import successful")
        except ImportError as e:
            print(f"⚠ SEKI package import failed: {e}")
            print("  This is expected if dependencies are missing")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False


def main():
    """主設置函數"""
    print("="*60)
    print("SEKI Project Setup")
    print("="*60)
    
    # 創建項目結構
    print("\n1. Creating project structure...")
    create_project_structure()
    
    # 安裝依賴
    print("\n2. Installing dependencies...")
    if not install_requirements():
        print("Setup failed during package installation")
        return False
    
    # 設置環境
    print("\n3. Setting up environment...")
    setup_environment()
    
    # 驗證安裝
    print("\n4. Verifying installation...")
    if verify_installation():
        print("\n" + "="*60)
        print("✓ SEKI project setup completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run experiments: python experiments/circle_experiment.py")
        print("2. Start Jupyter: jupyter notebook scripts/demo_notebook.ipynb")
        print("3. View documentation: docs/README.md")
        return True
    else:
        print("\n" + "="*60)
        print("✗ Setup completed with warnings")
        print("="*60)
        print("Please check the error messages above and install missing packages manually")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
