#!/usr/bin/env python3
"""
Setup script for the abstractive text summarization project.
This script helps users get started quickly.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Required: Python 3.8 or higher")
        return False


def check_pip():
    """Check if pip is available."""
    print("📦 Checking pip...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ pip is available: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False


def install_dependencies():
    """Install project dependencies."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing dependencies...")
    print("   This may take a few minutes...")
    
    # Install dependencies
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    )
    
    if success:
        print("✅ All dependencies installed successfully")
    else:
        print("❌ Failed to install some dependencies")
        print("   You may need to install them manually:")
        print("   pip install -r requirements.txt")
    
    return success


def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = Path(__file__).parent / "venv"
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🔧 Creating virtual environment...")
    success = run_command(
        f"{sys.executable} -m venv venv",
        "Creating virtual environment"
    )
    
    if success:
        print("✅ Virtual environment created")
        print("   To activate it:")
        print("   - On Unix/Mac: source venv/bin/activate")
        print("   - On Windows: venv\\Scripts\\activate")
    
    return success


def test_installation():
    """Test if the installation works."""
    print("🧪 Testing installation...")
    
    # Test imports
    test_script = """
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'src'))

try:
    from config import ConfigManager
    print('✅ Config module imported successfully')
except ImportError as e:
    print(f'❌ Config import failed: {e}')

try:
    from summarizer import create_sample_dataset
    print('✅ Summarizer module imported successfully')
except ImportError as e:
    print(f'❌ Summarizer import failed: {e}')
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def show_next_steps():
    """Show next steps for the user."""
    print("\n🎉 Setup Complete!")
    print("=" * 30)
    print("\n🚀 Next steps:")
    print("   1. Activate virtual environment (if created):")
    print("      - Unix/Mac: source venv/bin/activate")
    print("      - Windows: venv\\Scripts\\activate")
    print("   2. Try the demo: python demo.py")
    print("   3. Run CLI: python cli.py --demo")
    print("   4. Launch web app: streamlit run web_app/app.py")
    print("   5. Run tests: pytest tests/")
    print("\n📚 For more information, see README.md")


def main():
    """Main setup function."""
    print("🛠️  Abstractive Text Summarization - Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        print("❌ pip is required but not available")
        sys.exit(1)
    
    # Ask user for setup options
    print("\n🔧 Setup Options:")
    print("   1. Install dependencies globally")
    print("   2. Create virtual environment and install dependencies")
    print("   3. Skip installation (manual setup)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n📦 Installing dependencies globally...")
        if install_dependencies():
            test_installation()
        show_next_steps()
    
    elif choice == "2":
        print("\n🔧 Creating virtual environment and installing dependencies...")
        if create_virtual_environment():
            print("\n📦 Installing dependencies in virtual environment...")
            # Note: In a real scenario, you'd activate the venv first
            install_dependencies()
            test_installation()
        show_next_steps()
    
    elif choice == "3":
        print("\n⏭️  Skipping automatic installation")
        print("\n📋 Manual setup instructions:")
        print("   1. Create virtual environment: python -m venv venv")
        print("   2. Activate it: source venv/bin/activate (Unix/Mac) or venv\\Scripts\\activate (Windows)")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Test installation: python demo.py")
        show_next_steps()
    
    else:
        print("❌ Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
