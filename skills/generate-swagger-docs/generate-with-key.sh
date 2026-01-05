#!/bin/bash

# Generate Swagger Documentation with OpenAI API Key
# This script accepts an OpenAI API key and generates Swagger/OpenAPI docs

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}Swagger Documentation Generator${NC}"
echo -e "${BLUE}===================================================${NC}"
echo ""

# Check if API key is provided as argument
if [ -z "$1" ]; then
    echo -e "${RED}Error: OpenAI API key required${NC}"
    echo ""
    echo "Usage: $0 <openai-api-key>"
    echo ""
    echo "Example:"
    echo "  $0 sk-proj-your-api-key-here"
    echo ""
    echo "Or set the environment variable:"
    echo "  export OPENAI_API_KEY='sk-proj-your-api-key-here'"
    echo "  $0"
    exit 1
fi

API_KEY="$1"

# Validate API key format
if [[ ! "$API_KEY" =~ ^sk-proj- ]]; then
    echo -e "${RED}Warning: API key does not match expected format (should start with 'sk-proj-')${NC}"
    echo ""
fi

# Export the API key as environment variable
export OPENAI_API_KEY="$API_KEY"

echo -e "${GREEN}✓ OpenAI API key configured${NC}"
echo ""

# Create apimesh directory structure
echo -e "${BLUE}Setting up apimesh...${NC}"
mkdir -p apimesh

# Download the apimesh tool
echo -e "${BLUE}Downloading apimesh tool...${NC}"
curl -sSL https://raw.githubusercontent.com/qodex-ai/apimesh/refs/heads/main/run.sh -o apimesh/run.sh
chmod +x apimesh/run.sh

echo -e "${GREEN}✓ apimesh downloaded and configured${NC}"
echo ""

# Run the apimesh tool
echo -e "${BLUE}Generating Swagger documentation...${NC}"
echo -e "${BLUE}This may take a few moments...${NC}"
echo ""

# Store current directory (the repository to document)
REPO_DIR="$(pwd)"

# Pass the API key as a command-line argument to the apimesh run.sh script
# Run from the repository directory, not from inside apimesh folder
cd "$REPO_DIR"
bash apimesh/run.sh --openai-api-key "$API_KEY"

echo ""
echo -e "${GREEN}===================================================${NC}"
echo -e "${GREEN}✓ Swagger documentation generated successfully!${NC}"
echo -e "${GREEN}===================================================${NC}"
echo ""
echo "Output files:"
echo "  - apimesh/swagger.json"
echo "  - apimesh/apimesh-docs.html"
echo "  - apimesh/config.json (contains your settings)"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Add 'apimesh/config.json' to .gitignore"
echo "  2. Open apimesh/apimesh-docs.html in your browser to view docs"
echo "  3. Share apimesh-docs.html with your team"
echo ""
