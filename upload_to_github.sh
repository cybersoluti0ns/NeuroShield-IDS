#!/bin/bash

echo "🚀 NeuroShield IDS - GitHub Upload Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the NeuroShield IDS directory"
    exit 1
fi

echo "✅ Found NeuroShield IDS project files"

# Check git status
echo "📊 Git Status:"
git status --short

echo ""
echo "🔐 Authentication Options:"
echo "1. Personal Access Token (Recommended)"
echo "2. SSH Key"
echo "3. GitHub CLI"
echo ""

echo "📋 To upload your project:"
echo ""
echo "Option 1 - Personal Access Token:"
echo "1. Go to: https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Name: 'NeuroShield IDS Upload'"
echo "4. Select 'repo' scope"
echo "5. Copy the token"
echo "6. Run: git push https://YOUR_TOKEN@github.com/cybersoluti0ns/NeuroShield-IDS.git main"
echo ""

echo "Option 2 - SSH (if you have SSH keys):"
echo "1. Run: git remote set-url origin git@github.com:cybersoluti0ns/NeuroShield-IDS.git"
echo "2. Run: git push origin main"
echo ""

echo "Option 3 - GitHub CLI:"
echo "1. Install: curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg"
echo "2. Run: gh auth login"
echo "3. Run: git push origin main"
echo ""

echo "📁 Your project includes:"
echo "✅ Complete ML pipeline (99.55% accuracy)"
echo "✅ Interactive Streamlit dashboard (GUI)"
echo "✅ Trained models and datasets"
echo "✅ Comprehensive documentation"
echo "✅ Cybersecurity-focused features"
echo ""

echo "🎯 Repository URL: https://github.com/cybersoluti0ns/NeuroShield-IDS"
echo ""

echo "Ready to upload! Choose your preferred authentication method above."
