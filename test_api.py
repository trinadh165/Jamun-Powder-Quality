"""
Test script for the Jamun Powder Quality Evaluator API.
Tests various input scenarios and validates the output.
"""

import json
import requests
from utils import validate_input_structure

def test_validation():
    """Test input validation functionality."""
    print("Testing input validation...")
    
    # Test valid input
    valid_input = {
        "raw_material": {
            "ripeness": "ripe",
            "quality_uniformity": True
        },
        "freeze_drying": {
            "temperature": -40,
            "time_hours": 20
        },
        "packaging": {
            "type": "vacuum_pouch",
            "opaque": True,
            "airtight": True
        },
        "storage": {
            "temperature": "frozen",
            "humidity_percent": 5,
            "light_exposure": "dark"
        }
    }
    
    is_valid, error = validate_input_structure(valid_input)
    assert is_valid, f"Valid input should pass validation: {error}"
    print("✓ Valid input passes validation")
    
    # Test invalid input (missing field)
    invalid_input = {
        "raw_material": {
            "ripeness": "ripe"
            # Missing quality_uniformity
        },
        "freeze_drying": {
            "temperature": -40,
            "time_hours": 20
        },
        "packaging": {
            "type": "vacuum_pouch",
            "opaque": True,
            "airtight": True
        },
        "storage": {
            "temperature": "frozen",
            "humidity_percent": 5,
            "light_exposure": "dark"
        }
    }
    
    is_valid, error = validate_input_structure(invalid_input)
    assert not is_valid, "Invalid input should fail validation"
    print("✓ Invalid input fails validation")

def test_rules_engine():
    """Test the rules engine directly."""
    print("\nTesting rules engine...")
    
    from rules import evaluate_jamun_powder_quality
    
    test_input = {
        "raw_material": {
            "ripeness": "ripe",
            "quality_uniformity": True
        },
        "freeze_drying": {
            "temperature": -40,
            "time_hours": 20
        },
        "packaging": {
            "type": "vacuum_pouch",
            "opaque": True,
            "airtight": True
        },
        "storage": {
            "temperature": "frozen",
            "humidity_percent": 5,
            "light_exposure": "dark"
        }
    }
    
    results = evaluate_jamun_powder_quality(test_input)
    
    # Validate output structure
    assert "nutrition" in results
    assert "physical" in results
    assert "chemical" in results
    assert "shelf_life" in results
    
    # Validate nutrition scores
    assert "vitamin_c_score" in results["nutrition"]
    assert "antioxidant_score" in results["nutrition"]
    assert 0 <= results["nutrition"]["vitamin_c_score"] <= 100
    assert 0 <= results["nutrition"]["antioxidant_score"] <= 100
    
    # Validate physical scores
    assert "color_quality" in results["physical"]
    assert "powder_flow" in results["physical"]
    
    # Validate chemical scores
    assert "moisture_stability" in results["chemical"]
    
    # Validate shelf life
    assert "estimated_months" in results["shelf_life"]
    assert 1 <= results["shelf_life"]["estimated_months"] <= 24
    
    print("✓ Rules engine produces valid output structure")
    print(f"✓ Sample results: {json.dumps(results, indent=2)}")

def test_scenarios():
    """Test different input scenarios."""
    print("\nTesting different scenarios...")
    
    from rules import evaluate_jamun_powder_quality
    
    scenarios = [
        {
            "name": "Optimal conditions",
            "input": {
                "raw_material": {"ripeness": "ripe", "quality_uniformity": True},
                "freeze_drying": {"temperature": -40, "time_hours": 20},
                "packaging": {"type": "vacuum_pouch", "opaque": True, "airtight": True},
                "storage": {"temperature": "frozen", "humidity_percent": 5, "light_exposure": "dark"}
            }
        },
        {
            "name": "Poor conditions",
            "input": {
                "raw_material": {"ripeness": "unripe", "quality_uniformity": False},
                "freeze_drying": {"temperature": -10, "time_hours": 35},
                "packaging": {"type": "normal_pouch", "opaque": False, "airtight": False},
                "storage": {"temperature": "room", "humidity_percent": 25, "light_exposure": "light"}
            }
        },
        {
            "name": "Mixed conditions",
            "input": {
                "raw_material": {"ripeness": "overripe", "quality_uniformity": True},
                "freeze_drying": {"temperature": -30, "time_hours": 25},
                "packaging": {"type": "glass_jar", "opaque": True, "airtight": True},
                "storage": {"temperature": "refrigerated", "humidity_percent": 8, "light_exposure": "dark"}
            }
        }
    ]
    
    for scenario in scenarios:
        results = evaluate_jamun_powder_quality(scenario["input"])
        print(f"✓ {scenario['name']}: Vitamin C = {results['nutrition']['vitamin_c_score']}, "
              f"Shelf life = {results['shelf_life']['estimated_months']} months")

if __name__ == "__main__":
    print("Running Jamun Powder Quality Evaluator Tests\n")
    print("=" * 50)
    
    try:
        test_validation()
        test_rules_engine()
        test_scenarios()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("\nThe system is ready for use.")
        print("Start the Flask server with: python app.py")
        print("Then send POST requests to http://localhost:5000/evaluate")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
