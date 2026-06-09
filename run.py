#!/usr/bin/env python3
"""
Lung Cancer Diagnostic System - Main Entry Point
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'integrated_lung_cancer_system'))


def main():
    """Main application entry point"""
    from app import app

    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'true').lower() in ('1', 'true', 'yes')

    print("=" * 60)
    print("Lung Cancer Diagnostic System")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
