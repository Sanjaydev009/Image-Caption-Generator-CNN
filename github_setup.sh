#!/bin/bash
# Script to setup and push the project to GitHub

echo "🚀 Setting up GitHub repository for Image Caption Generator"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
fi

# Ask for GitHub repo details
echo "📝 Enter your GitHub username:"
read github_username

echo "📝 Enter your repository name (e.g., image-caption-generator):"
read repo_name

# Add all files
echo "📁 Adding files to Git..."
git add .

# Commit changes
echo "💾 Committing files..."
git commit -m "Initial commit: Image Caption Generator with translation feature"

# Add remote repository
echo "🔗 Adding remote repository..."
git remote add origin https://github.com/$github_username/$repo_name.git

# Determine default branch
default_branch=$(git symbolic-ref --short HEAD)

# Push to GitHub
echo "⬆️ Pushing to GitHub..."
git push -u origin $default_branch

echo "✅ Done! Your project is now on GitHub at: https://github.com/$github_username/$repo_name"
