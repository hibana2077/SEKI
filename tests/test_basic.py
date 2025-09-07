"""
SEKI 基礎測試
驗證核心功能是否正常工作
"""

import unittest
import numpy as np
import sys
import os

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestSEKIBasics(unittest.TestCase):
    """SEKI 基礎功能測試"""
    
    def setUp(self):
        """測試設置"""
        self.test_data_size = 50
        self.input_dim = 2
        
    def test_numpy_available(self):
        """測試 NumPy 是否可用"""
        try:
            import numpy as np
            self.assertTrue(True, "NumPy is available")
            
            # 測試基本操作
            arr = np.random.randn(10, 2)
            self.assertEqual(arr.shape, (10, 2))
            
        except ImportError:
            self.fail("NumPy is not available")
    
    def test_torch_available(self):
        """測試 PyTorch 是否可用"""
        try:
            import torch
            self.assertTrue(True, "PyTorch is available")
            
            # 測試基本操作
            tensor = torch.randn(10, 2)
            self.assertEqual(tensor.shape, (10, 2))
            
        except ImportError:
            self.fail("PyTorch is not available")
    
    def test_sklearn_available(self):
        """測試 scikit-learn 是否可用"""
        try:
            from sklearn.datasets import make_circles
            from sklearn.preprocessing import StandardScaler
            self.assertTrue(True, "scikit-learn is available")
            
            # 測試基本操作
            X, y = make_circles(n_samples=100, noise=0.1, random_state=42)
            self.assertEqual(X.shape, (100, 2))
            self.assertEqual(y.shape, (100,))
            
        except ImportError:
            self.fail("scikit-learn is not available")
    
    def test_seki_package_import(self):
        """測試 SEKI 包是否可以導入"""
        try:
            import seki
            self.assertTrue(True, "SEKI package can be imported")
        except ImportError as e:
            self.skipTest(f"SEKI package not available: {e}")
    
    def test_seki_augmentation_functions(self):
        """測試 SEKI 增強函數"""
        try:
            import torch
            from seki.augmentation import seki_augmentation, mixup_augmentation
            
            # 創建測試數據
            X = torch.randn(20, 2)
            y = torch.randint(0, 2, (20,)).float()
            
            # 測試 SEKI 增強
            X_seki, y_seki = seki_augmentation(X, y, alpha=1.0)
            self.assertEqual(X_seki.shape, X.shape)
            self.assertEqual(y_seki.shape, y.shape)
            
            # 測試 Mixup 增強
            X_mixup, y_mixup = mixup_augmentation(X, y, alpha=1.0)
            self.assertEqual(X_mixup.shape, X.shape)
            self.assertEqual(y_mixup.shape, y.shape)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        except Exception as e:
            self.fail(f"Augmentation functions failed: {e}")
    
    def test_seki_models(self):
        """測試 SEKI 模型"""
        try:
            import torch
            from seki.models import StandardMLP
            
            # 創建模型
            model = StandardMLP(input_dim=2, hidden_dim=16, output_dim=1)
            self.assertIsNotNone(model)
            
            # 測試前向傳播
            X = torch.randn(10, 2)
            output = model(X)
            self.assertEqual(output.shape, (10, 1))
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        except Exception as e:
            self.fail(f"Model creation/forward pass failed: {e}")
    
    def test_polar_coordinate_conversion(self):
        """測試極坐標轉換"""
        try:
            import torch
            from seki.augmentation import to_polar, to_cartesian
            
            # 創建測試數據
            X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            
            # 轉換到極坐標
            r, theta = to_polar(X)
            
            # 檢查半徑
            expected_r = torch.ones(4)
            torch.testing.assert_close(r, expected_r, atol=1e-6, rtol=1e-6)
            
            # 轉換回笛卡爾坐標
            X_reconstructed = to_cartesian(r, theta)
            torch.testing.assert_close(X_reconstructed, X, atol=1e-6, rtol=1e-6)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        except Exception as e:
            self.fail(f"Polar coordinate conversion failed: {e}")
    
    def test_project_structure(self):
        """測試項目結構"""
        required_dirs = ['seki', 'experiments', 'scripts', 'results']
        
        for directory in required_dirs:
            dir_path = os.path.join(project_root, directory)
            self.assertTrue(
                os.path.exists(dir_path), 
                f"Required directory {directory} does not exist"
            )
        
        required_files = [
            'seki/__init__.py',
            'seki/augmentation.py',
            'seki/models.py',
            'seki/evaluation.py',
            'seki/training.py',
            'requirements.txt',
            'README.md'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(project_root, file_path)
            self.assertTrue(
                os.path.exists(full_path),
                f"Required file {file_path} does not exist"
            )


class TestSEKIIntegration(unittest.TestCase):
    """SEKI 集成測試"""
    
    def setUp(self):
        """測試設置"""
        try:
            import torch
            from sklearn.datasets import make_circles
            
            # 生成測試數據
            X_np, y_np = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
            self.X = torch.FloatTensor(X_np)
            self.y = torch.FloatTensor(y_np)
            
        except ImportError:
            self.skipTest("Required dependencies not available")
    
    def test_end_to_end_workflow(self):
        """測試端到端工作流程"""
        try:
            import torch
            from seki import (
                seki_augmentation,
                StandardMLP,
                evaluate_accuracy
            )
            
            # 數據增強
            X_aug, y_aug = seki_augmentation(self.X, self.y, alpha=0.5)
            self.assertEqual(X_aug.shape, self.X.shape)
            
            # 創建和訓練簡單模型
            model = StandardMLP(input_dim=2, hidden_dim=16, output_dim=1)
            
            # 簡單訓練步驟
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for _ in range(10):  # 只訓練幾步
                optimizer.zero_grad()
                output = model(self.X).squeeze()
                loss = criterion(output, self.y)
                loss.backward()
                optimizer.step()
            
            # 評估
            accuracy, _ = evaluate_accuracy(model, self.X, self.y)
            self.assertIsInstance(accuracy, float)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        except Exception as e:
            self.fail(f"End-to-end workflow failed: {e}")


def run_tests():
    """運行所有測試"""
    print("Running SEKI Tests...")
    print("=" * 50)
    
    # 創建測試套件
    suite = unittest.TestSuite()
    
    # 添加基礎測試
    suite.addTest(unittest.makeSuite(TestSEKIBasics))
    suite.addTest(unittest.makeSuite(TestSEKIIntegration))
    
    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印結果摘要
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
