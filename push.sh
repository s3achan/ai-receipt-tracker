#!/bin/bash


# ── Add, commit and push ───────────────────────────────────────────────────
# Usage: bash push.sh "your commit message"

MESSAGE=${1:-"Update files"}

echo "📦 Staging all changes..."
git add .

echo "✅ Committing: $MESSAGE"
git commit -m "$MESSAGE"

echo "🚀 Pushing..."
git push

echo ""
echo "🎉 Done!"