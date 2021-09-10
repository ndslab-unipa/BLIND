#!/bin/bash

pytest --cov-report html:blind/test/htmlcov blind/test/test_service_provider.py --cov=blind
