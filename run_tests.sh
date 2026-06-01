# Activate venv and run all tests
source venv/bin/activate
pytest  # or: python -m pytest

# Run specific module
python -m pytest tests/test_cli.py -v

# Show skipped test reasons
python -m pytest --reason

read -p "Press return (ENTER) to continue and exit this script... " CONTINUE_KEY
