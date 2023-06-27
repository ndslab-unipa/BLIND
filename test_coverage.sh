#!/bin/bash

# Generate the pytest coverage report

pytest --cov-report html:blind/test/htmlcov blind/test/test_service_provider.py --cov=blind
