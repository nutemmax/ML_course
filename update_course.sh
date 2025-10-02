#!/bin/bash
set -euo pipefail

echo ">>> Updating your ML_course fork"

cd ~/ML_course

echo ">>> Switching to main branch..."
git checkout main

echo ">>> Fetching latest changes from EPFL upstream..."
git fetch upstream

echo ">>> Merging upstream/main into your local main..."
git merge upstream/main

echo ">>> Pushing updated main to your fork (origin)..."
git push origin main

echo "✅ Update complete!"
