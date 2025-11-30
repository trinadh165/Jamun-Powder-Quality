"""
Utility functions for scoring and calculations in Jamun powder quality evaluation.
Contains helper functions for normalizing scores, applying penalties, and calculating final values.
"""

def clamp_score(value, min_val=0, max_val=100):
    """
    Clamp a score value within the specified range.
    
    Args:
        value (float): The score to clamp
        min_val (int): Minimum allowed value
        max_val (int): Maximum allowed value
    
    Returns:
        int: Clamped score
    """
    return max(min_val, min(max_val, int(value)))

def apply_percentage_penalty(base_score, penalty_percent):
    """
    Apply a percentage penalty to a base score.
    
    Args:
        base_score (float): Original score
        penalty_percent (float): Penalty percentage (e.g., 10 for 10% penalty)
    
    Returns:
        int: Score after penalty
    """
    return clamp_score(base_score * (1 - penalty_percent / 100))

def calculate_weighted_average(scores, weights):
    """
    Calculate weighted average of multiple scores.
    
    Args:
        scores (list): List of scores
        weights (list): List of corresponding weights
    
    Returns:
        int: Weighted average score
    """
    if not scores or not weights or len(scores) != len(weights):
        return 0
    
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    
    return clamp_score(weighted_sum / total_weight if total_weight > 0 else 0)

def get_quality_category(score):
    """
    Convert numeric score to quality category.
    
    Args:
        score (int): Numeric score (0-100)
    
    Returns:
        str: Quality category
    """
    if score >= 90:
        return "excellent"
    elif score >= 80:
        return "very_good"
    elif score >= 70:
        return "good"
    elif score >= 60:
        return "fair"
    elif score >= 50:
        return "poor"
    else:
        return "very_poor"

def get_flow_quality(score):
    """
    Convert score to powder flow quality description.
    
    Args:
        score (int): Numeric score (0-100)
    
    Returns:
        str: Flow quality description
    """
    if score >= 85:
        return "excellent"
    elif score >= 70:
        return "good"
    elif score >= 55:
        return "moderate"
    elif score >= 40:
        return "poor"
    else:
        return "very_poor"

def get_stability_status(score):
    """
    Convert score to stability status.
    
    Args:
        score (int): Numeric score (0-100)
    
    Returns:
        str: Stability status
    """
    if score >= 75:
        return "stable"
    elif score >= 50:
        return "moderately_stable"
    else:
        return "unstable"

def estimate_shelf_life_months(nutrition_score, physical_score, chemical_score):
    """
    Estimate shelf life in months based on quality scores.
    
    Args:
        nutrition_score (int): Overall nutrition score
        physical_score (int): Overall physical quality score
        chemical_score (int): Overall chemical stability score
    
    Returns:
        int: Estimated shelf life in months
    """
    # Base shelf life calculation
    base_months = 12
    
    # Adjust based on scores
    nutrition_factor = nutrition_score / 100
    physical_factor = physical_score / 100
    chemical_factor = chemical_score / 100
    
    # Calculate weighted shelf life
    shelf_life = base_months * (
        nutrition_factor * 0.4 + 
        physical_factor * 0.3 + 
        chemical_factor * 0.3
    )
    
    # Ensure minimum and maximum shelf life
    return max(1, min(24, int(shelf_life)))

def validate_input_structure(data):
    """
    Validate the input JSON structure.
    
    Args:
        data (dict): Input data to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    required_sections = ["raw_material", "freeze_drying", "packaging", "storage"]
    
    # Check top-level sections
    for section in required_sections:
        if section not in data:
            return False, f"Missing required section: {section}"
    
    # Check raw_material fields
    raw_material = data["raw_material"]
    if "ripeness" not in raw_material:
        return False, "Missing raw_material.ripeness"
    if "quality_uniformity" not in raw_material:
        return False, "Missing raw_material.quality_uniformity"
    
    # Check freeze_drying fields
    freeze_drying = data["freeze_drying"]
    if "temperature" not in freeze_drying:
        return False, "Missing freeze_drying.temperature"
    if "time_hours" not in freeze_drying:
        return False, "Missing freeze_drying.time_hours"
    
    # Check packaging fields
    packaging = data["packaging"]
    if "type" not in packaging:
        return False, "Missing packaging.type"
    if "opaque" not in packaging:
        return False, "Missing packaging.opaque"
    if "airtight" not in packaging:
        return False, "Missing packaging.airtight"
    
    # Check storage fields
    storage = data["storage"]
    if "temperature" not in storage:
        return False, "Missing storage.temperature"
    if "humidity_percent" not in storage:
        return False, "Missing storage.humidity_percent"
    if "light_exposure" not in storage:
        return False, "Missing storage.light_exposure"
    
    return True, ""
