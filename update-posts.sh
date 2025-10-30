#!/bin/bash
# Automatically generate posts.json from all .md files in thoughts/

cd "$(dirname "$0")"

echo "Generating thoughts/posts.json..."

# Find all .md files in thoughts/, sort by modification time (newest first), output as JSON array
cd thoughts
ls -t *.md 2>/dev/null | jq -R -s -c 'split("\n") | map(select(length > 0))' > posts.json

echo "Done! Found $(cat posts.json | jq 'length') posts."
