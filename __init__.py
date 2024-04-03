import sys
import os

# Get the directory of your current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)