#!/usr/bin/env python3
"""
TRU Salary Predictor - Presentation Runner
==========================================

This script helps you run and display your presentation content.
It can show the presentation slides, open the app, and provide guidance.

Author: TRU Salary Predictor
Email: truptibhuskute@gmail.com
Copyright © 2025 TRU Salary Predictor
"""

import os
import webbrowser
import subprocess
import sys
from pathlib import Path

class PresentationRunner:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.presentation_file = self.project_dir / "TRU_Salary_Predictor_Presentation.md"
        self.template_file = self.project_dir / "TRU_Salary_Predictor_PPT_Template.md"
        self.guide_file = self.project_dir / "PRESENTATION_DOWNLOAD_GUIDE.md"
        
    def show_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🎯 TRU SALARY PREDICTOR - PRESENTATION RUNNER")
        print("="*60)
        print("1. 📊 View Presentation Content (21 slides)")
        print("2. 🎨 View PowerPoint Template")
        print("3. 📥 View Download Guide")
        print("4. 🌐 Open Live App (Streamlit)")
        print("5. 🔗 Open GitHub Repository")
        print("6. 📋 Show All Available Files")
        print("7. 🚀 Quick Start Guide")
        print("0. ❌ Exit")
        print("="*60)
        
    def view_presentation_content(self):
        """Display the presentation content"""
        if self.presentation_file.exists():
            print("\n📊 PRESENTATION CONTENT (21 SLIDES)")
            print("="*50)
            with open(self.presentation_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        else:
            print("❌ Presentation file not found!")
            
    def view_template(self):
        """Display the PowerPoint template"""
        if self.template_file.exists():
            print("\n🎨 POWERPOINT TEMPLATE")
            print("="*50)
            with open(self.template_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        else:
            print("❌ Template file not found!")
            
    def view_guide(self):
        """Display the download guide"""
        if self.guide_file.exists():
            print("\n📥 DOWNLOAD GUIDE")
            print("="*50)
            with open(self.guide_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        else:
            print("❌ Guide file not found!")
            
    def open_live_app(self):
        """Open the live Streamlit app"""
        try:
            webbrowser.open("http://localhost:8501")
            print("✅ Opening TRU Salary Predictor app...")
            print("🌐 URL: http://localhost:8501")
        except Exception as e:
            print(f"❌ Error opening app: {e}")
            
    def open_github(self):
        """Open the GitHub repository"""
        try:
            webbrowser.open("https://github.com/truptibhuskute/tru-salary-predictor")
            print("✅ Opening GitHub repository...")
        except Exception as e:
            print(f"❌ Error opening GitHub: {e}")
            
    def show_files(self):
        """Show all available files"""
        print("\n📋 AVAILABLE FILES IN PROJECT")
        print("="*50)
        files = list(self.project_dir.glob("*"))
        for file in sorted(files):
            if file.is_file():
                size = file.stat().st_size
                print(f"📄 {file.name} ({size:,} bytes)")
            else:
                print(f"📁 {file.name}/")
                
    def quick_start_guide(self):
        """Show quick start guide"""
        print("\n🚀 QUICK START GUIDE")
        print("="*50)
        print("1. 📊 Your app is running at: http://localhost:8501")
        print("2. 📋 Presentation files are ready for download")
        print("3. 🎯 Copy content from markdown files to PowerPoint")
        print("4. 🎨 Follow the template guidelines for design")
        print("5. 📱 Add screenshots of your live app")
        print("6. 🎉 Practice your presentation!")
        print("\n📧 Contact: truptibhuskute@gmail.com")
        print("🔗 GitHub: https://github.com/truptibhuskute/tru-salary-predictor")
        
    def run(self):
        """Main run loop"""
        while True:
            self.show_menu()
            try:
                choice = input("\n🎯 Enter your choice (0-7): ").strip()
                
                if choice == "1":
                    self.view_presentation_content()
                elif choice == "2":
                    self.view_template()
                elif choice == "3":
                    self.view_guide()
                elif choice == "4":
                    self.open_live_app()
                elif choice == "5":
                    self.open_github()
                elif choice == "6":
                    self.show_files()
                elif choice == "7":
                    self.quick_start_guide()
                elif choice == "0":
                    print("\n👋 Thank you for using TRU Salary Predictor!")
                    print("🎉 Good luck with your presentation!")
                    break
                else:
                    print("❌ Invalid choice! Please enter 0-7.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
            input("\n⏸️  Press Enter to continue...")

def main():
    """Main function"""
    print("🎯 TRU Salary Predictor - Presentation Runner")
    print("📧 Contact: truptibhuskute@gmail.com")
    print("🔗 GitHub: https://github.com/truptibhuskute/tru-salary-predictor")
    
    runner = PresentationRunner()
    runner.run()

if __name__ == "__main__":
    main() 