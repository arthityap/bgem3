#!/bin/zsh
# Script to remove old virtual environments
# Run with sudo: sudo ./remove_old_venvs.sh

echo "Removing old virtual environments..."

# Remove bgem3-venv if it exists
if [ -d "bgem3-venv" ]; then
    rm -rf bgem3-venv
    echo "Removed bgem3-venv"
else
    echo "bgem3-venv not found"
fi

# Remove .venv if it exists
if [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed .venv"
else
    echo ".venv not found"
fi

echo "Cleanup complete"