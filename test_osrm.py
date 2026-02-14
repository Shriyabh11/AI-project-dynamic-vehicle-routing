"""
Quick OSRM Test - Check if getting real road curves
"""

import sys
sys.path.append('dynamic_routing')

from traffic_api import TrafficAPI

def test_osrm():
    """Test OSRM geometry - should return 20+ points"""
    api = TrafficAPI()
    
    # Test route in Bangalore
    start = (12.9716, 77.5946)  # Bangalore center
    end = (12.9352, 77.6245)    # Koramangala
    
    print("Testing OSRM Route Geometry...")
    print(f"From: {start}")
    print(f"To: {end}")
    print()
    
    result = api.get_route_geometry([start, end])
    
    print(f"✓ Points returned: {len(result)}")
    print(f"✓ First 3 points: {result[:3]}")
    print(f"✓ Last 3 points: {result[-3:]}")
    print()
    
    if len(result) > 10:
        print("✅ SUCCESS - OSRM returning real curved roads!")
        print(f"   ({len(result)} coordinate points = smooth curves)")
    elif len(result) > 2:
        print("⚠️  PARTIAL - Some geometry but may not be fully curved")
    else:
        print("❌ FAIL - Only straight line (2 points)")
        print("   OSRM may be blocked or not responding")
    
    return len(result) > 10

if __name__ == "__main__":
    success = test_osrm()
    sys.exit(0 if success else 1)
