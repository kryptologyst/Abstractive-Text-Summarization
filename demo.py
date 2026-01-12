#!/usr/bin/env python3
"""
Simple demo script for the abstractive text summarization project.
This script demonstrates the project structure and basic functionality without heavy dependencies.
"""

import sys
from pathlib import Path

def show_project_structure():
    """Show the project structure."""
    print("📁 Project Structure")
    print("=" * 25)
    
    project_root = Path(__file__).parent
    
    structure = [
        ("src/", "Core source code"),
        ("  ├── summarizer.py", "Main summarization module"),
        ("  ├── config.py", "Configuration management"),
        ("  └── visualization.py", "Visualization tools"),
        ("web_app/", "Streamlit web interface"),
        ("  └── app.py", "Main web application"),
        ("tests/", "Test suite"),
        ("  └── test_summarizer.py", "Unit tests"),
        ("config/", "Configuration files"),
        ("  └── config.yaml", "Default configuration"),
        ("data/", "Sample data and datasets"),
        ("models/", "Saved models (optional)"),
        ("cli.py", "Command-line interface"),
        ("requirements.txt", "Python dependencies"),
        (".gitignore", "Git ignore rules"),
        ("README.md", "Project documentation"),
        ("demo.py", "This demo script")
    ]
    
    for path, description in structure:
        print(f"   {path:<20} {description}")


def check_file_structure():
    """Check if the project files exist."""
    print("\n🔍 Checking Project Files")
    print("=" * 30)
    
    project_root = Path(__file__).parent
    required_files = [
        "src/summarizer.py",
        "src/config.py", 
        "src/visualization.py",
        "web_app/app.py",
        "tests/test_summarizer.py",
        "config/config.yaml",
        "cli.py",
        "requirements.txt",
        ".gitignore",
        "README.md"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            existing_files.append(file_path)
            print(f"   ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   ❌ {file_path}")
    
    print(f"\n📊 File Status: {len(existing_files)}/{len(required_files)} files exist")
    
    if missing_files:
        print(f"⚠️  Missing files: {', '.join(missing_files)}")
    
    return len(missing_files) == 0


def show_sample_config():
    """Show sample configuration."""
    print("\n⚙️  Sample Configuration")
    print("=" * 30)
    
    config_path = Path(__file__).parent / "config" / "config.yaml"
    
    if config_path.exists():
        print("✅ Configuration file exists")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                print(f"📄 Config file size: {len(content)} characters")
                print("📋 Configuration preview:")
                lines = content.split('\n')[:10]  # Show first 10 lines
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
                if len(content.split('\n')) > 10:
                    print("   ...")
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    else:
        print("❌ Configuration file not found")


def show_requirements():
    """Show requirements information."""
    print("\n📦 Requirements")
    print("=" * 20)
    
    req_path = Path(__file__).parent / "requirements.txt"
    
    if req_path.exists():
        print("✅ Requirements file exists")
        try:
            with open(req_path, 'r') as f:
                lines = f.readlines()
                print(f"📄 {len(lines)} dependencies listed")
                print("📋 Key dependencies:")
                key_deps = ["torch", "transformers", "streamlit", "datasets", "evaluate"]
                for dep in key_deps:
                    for line in lines:
                        if dep in line.lower() and not line.startswith('#'):
                            print(f"   {line.strip()}")
                            break
        except Exception as e:
            print(f"❌ Error reading requirements: {e}")
    else:
        print("❌ Requirements file not found")


def show_usage_examples():
    """Show usage examples."""
    print("\n🚀 Usage Examples")
    print("=" * 20)
    
    examples = [
        ("Install dependencies", "pip install -r requirements.txt"),
        ("Run CLI demo", "python cli.py --demo"),
        ("Summarize text", "python cli.py --text 'Your text here'"),
        ("Launch web app", "streamlit run web_app/app.py"),
        ("Run tests", "pytest tests/"),
        ("Create config", "python src/config.py")
    ]
    
    for description, command in examples:
        print(f"   {description:<20} {command}")


def main():
    """Main demo function."""
    print("🎯 Abstractive Text Summarization - Project Overview")
    print("=" * 60)
    
    # Show project structure
    show_project_structure()
    
    # Check files
    files_ok = check_file_structure()
    
    # Show configuration
    show_sample_config()
    
    # Show requirements
    show_requirements()
    
    # Show usage examples
    show_usage_examples()
    
    # Summary
    print(f"\n📈 Project Status")
    print("=" * 20)
    
    if files_ok:
        print("✅ Project structure is complete!")
        print("\n🎉 Ready to use! Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Try CLI: python cli.py --demo")
        print("   3. Launch web app: streamlit run web_app/app.py")
        print("   4. Run tests: pytest tests/")
    else:
        print("⚠️  Some files are missing. Check the file structure above.")
    
    print("\n📚 For detailed information, see README.md")
    print("🔧 For configuration options, see config/config.yaml")


if __name__ == "__main__":
    main()