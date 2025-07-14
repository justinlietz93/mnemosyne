import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aegis_layer import Aegis

# Test validate_response with safe response
safe_response = "This is a safe response."
if Aegis().validate_response(safe_response):
    print("Safe response validated correctly.")
else:
    print("Error: Safe response not validated.")

# Test validate_response with unsafe response
unsafe_response = "This is harmful content."
if not Aegis().validate_response(unsafe_response):
    print("Unsafe response detected correctly.")
else:
    print("Error: Unsafe response not detected.")

# Additional test with another unsafe keyword
dangerous_response = "This is dangerous."
if not Aegis().validate_response(dangerous_response):
    print("Dangerous response detected correctly.")
else:
    print("Error: Dangerous response not detected.")

print("Aegis test completed. Check for any errors above.")