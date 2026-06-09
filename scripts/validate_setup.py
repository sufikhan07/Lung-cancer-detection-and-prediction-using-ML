#!/usr/bin/env python3
"""
Setup validation script for Lung Cancer Diagnostic System
"""
import os
import sys
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.9+")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    required_packages = [
        'flask', 'tensorflow', 'numpy', 'PIL', 'cv2', 'sklearn', 'pandas'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)

    return len(missing_packages) == 0

def check_models():
    """Check if ML models are available"""
    print("\n🧠 Checking ML models...")

    model_paths = [
        'models/lung_cancer_cnn_model.keras',
        'models/risk_assessment_model.h5',
        'models/scaler.pkl'
    ]

    missing_models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ {model_path} - FOUND")
        else:
            print(f"❌ {model_path} - MISSING")
            missing_models.append(model_path)

    return len(missing_models) == 0

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")

    required_dirs = [
        'static/uploads',
        'logs',
        'models'
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} - EXISTS")
        else:
            print(f"⚠️  {dir_path} - MISSING (will be created)")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ {dir_path} - CREATED")
            except Exception as e:
                print(f"❌ {dir_path} - FAILED TO CREATE: {e}")
                missing_dirs.append(dir_path)

    return len(missing_dirs) == 0

def check_configuration():
    """Check configuration files"""
    print("\n⚙️  Checking configuration...")

    config_checks = [
        ('.env', 'Environment configuration'),
        ('requirements.txt', 'Python dependencies'),
        ('config/__init__.py', 'Configuration module'),
    ]

    missing_configs = []
    for config_file, description in config_checks:
        if os.path.exists(config_file):
            print(f"✅ {config_file} ({description}) - FOUND")
        else:
            print(f"❌ {config_file} ({description}) - MISSING")
            missing_configs.append(config_file)

    return len(missing_configs) == 0

def check_database():
    """Check database connectivity (optional)"""
    print("\n🗄️  Checking database connectivity...")

    try:
        import psycopg2
        # Try to connect using environment variables
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'lung_cancer_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
        }

        conn = psycopg2.connect(**db_config)
        conn.close()
        print("✅ Database connection - OK")
        return True
    except ImportError:
        print("⚠️  psycopg2 not installed - Database check skipped")
        return True
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("   (This is optional - application will work without database)")
        return True

def main():
    """Run all validation checks"""
    print("🔍 Lung Cancer Diagnostic System - Setup Validation")
    print("=" * 60)

    checks = [
        check_python_version,
        check_dependencies,
        check_models,
        check_directories,
        check_configuration,
        check_database,
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Error during {check.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📊 Validation Summary:")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print("\n🚀 System is ready to run!")
        print("   Start with: python run.py")
        return 0
    else:
        print(f"❌ Some checks failed ({passed}/{total})")
        print("\n🔧 Please fix the issues above before running the application.")
        print("   See docs/deployment.md for detailed setup instructions.")
        return 1

if __name__ == '__main__':
    sys.exit(main())</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\scripts\validate_setup.py