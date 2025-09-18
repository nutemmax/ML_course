#!/bin/bash
set -e

cd ~/ML_course

echo ">>> Switching to main"
git checkout main

echo ">>> Fetching upstream..."
git fetch upstream

echo ">>> Merging upstream/main into local main"
git merge upstream/main

echo ">>> Pushing updated main to your fork"
git push origin main

echo ">>> If you want your exercise branch updated too, run:"
echo "    git checkout ex01-emma && git rebase main"
