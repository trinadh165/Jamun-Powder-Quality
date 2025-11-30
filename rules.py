"""
Rule-based evaluation logic for freeze-dried Jamun powder quality.
Contains all rule functions for different aspects of quality assessment.
"""

from utils import (
    clamp_score, apply_percentage_penalty, calculate_weighted_average,
    get_quality_category, get_flow_quality, get_stability_status,
    estimate_shelf_life_months
)

def evaluate_raw_material(raw_material):
    """
    Evaluate raw material quality based on ripeness and uniformity.
    
    Args:
        raw_material (dict): Raw material parameters
    
    Returns:
        dict: Raw material evaluation scores
    """
    ripeness = raw_material["ripeness"]
    uniformity = raw_material["quality_uniformity"]
    
    # Base scores for different ripeness levels
    ripeness_scores = {
        "ripe": {"vitamin_c": 90, "antioxidant": 85, "color": 88},
        "unripe": {"vitamin_c": 60, "antioxidant": 70, "color": 65},
        "overripe": {"vitamin_c": 75, "antioxidant": 80, "color": 70}
    }
    
    # Get base scores for current ripeness
    base_scores = ripeness_scores.get(ripeness, ripeness_scores["ripe"])
    
    # Apply uniformity penalty
    if not uniformity:
        # Non-uniform quality reduces overall quality by 15%
        penalty = 15
        for key in base_scores:
            base_scores[key] = apply_percentage_penalty(base_scores[key], penalty)
    
    return base_scores

def evaluate_freeze_drying(freeze_drying):
    """
    Evaluate freeze-drying process quality.
    
    Args:
        freeze_drying (dict): Freeze-drying parameters
    
    Returns:
        dict: Freeze-drying evaluation scores
    """
    temperature = freeze_drying["temperature"]
    time_hours = freeze_drying["time_hours"]
    
    # Temperature rules
    if temperature <= -40:
        # Best nutrient retention at ≤ -40°C
        temp_score = 95
        nutrient_retention = 0.95
    elif temperature <= -30:
        temp_score = 80
        nutrient_retention = 0.85
    elif temperature <= -20:
        temp_score = 60
        nutrient_retention = 0.70
    else:
        # Poor antioxidant retention > -20°C
        temp_score = 40
        nutrient_retention = 0.50
    
    # Time rules
    if time_hours <= 24:
        # Optimal drying time
        time_score = 90
        time_factor = 1.0
    elif time_hours <= 30:
        time_score = 75
        time_factor = 0.95
    else:
        # Nutrient loss after 30 hours
        time_score = 60
        time_factor = 0.85
    
    # Combined freeze-drying score
    overall_score = calculate_weighted_average([temp_score, time_score], [0.6, 0.4])
    
    return {
        "overall_score": overall_score,
        "nutrient_retention_factor": nutrient_retention * time_factor,
        "temperature_score": temp_score,
        "time_score": time_score
    }

def evaluate_packaging(packaging):
    """
    Evaluate packaging quality and protection.
    
    Args:
        packaging (dict): Packaging parameters
    
    Returns:
        dict: Packaging evaluation scores
    """
    packaging_type = packaging["type"]
    opaque = packaging["opaque"]
    airtight = packaging["airtight"]
    
    # Base scores for packaging types
    type_scores = {
        "vacuum_pouch": 85,
        "glass_jar": 90,
        "normal_pouch": 60
    }
    
    base_score = type_scores.get(packaging_type, 60)
    
    # Apply feature bonuses/penalties
    if opaque:
        # Opaque packaging protects from light
        base_score = min(100, base_score + 10)
    else:
        # Transparent packaging allows light exposure
        base_score = apply_percentage_penalty(base_score, 15)
    
    if airtight:
        # Airtight packaging prevents oxidation
        base_score = min(100, base_score + 10)
    else:
        # Non-airtight allows oxidation
        base_score = apply_percentage_penalty(base_score, 20)
    
    # Special bonus for vacuum + opaque + airtight combination
    if packaging_type == "vacuum_pouch" and opaque and airtight:
        base_score = min(100, base_score + 5)  # Excellent nutrient protection
    
    # Penalty for normal pouch
    if packaging_type == "normal_pouch":
        base_score = apply_percentage_penalty(base_score, 10)  # Oxidation penalty
    
    return {
        "protection_score": base_score,
        "oxidation_protection": "high" if airtight else "low",
        "light_protection": "high" if opaque else "low"
    }

def evaluate_storage(storage):
    """
    Evaluate storage conditions.
    
    Args:
        storage (dict): Storage parameters
    
    Returns:
        dict: Storage evaluation scores
    """
    temperature = storage["temperature"]
    humidity = storage["humidity_percent"]
    light_exposure = storage["light_exposure"]
    
    # Temperature scoring
    temp_scores = {
        "frozen": 95,
        "refrigerated": 85,
        "room": 70
    }
    temp_score = temp_scores.get(temperature, 70)
    
    # Humidity scoring
    if humidity <= 5:
        humidity_score = 95
    elif humidity <= 10:
        humidity_score = 80
    elif humidity <= 15:
        humidity_score = 60
    else:
        # High humidity causes moisture instability
        humidity_score = 40
    
    # Light exposure scoring
    if light_exposure == "dark":
        light_score = 90
    else:
        # Light exposure causes Vitamin C degradation
        light_score = 60
    
    # Overall storage score
    overall_score = calculate_weighted_average(
        [temp_score, humidity_score, light_score], 
        [0.4, 0.3, 0.3]
    )
    
    # Shelf life factor
    shelf_life_factors = {
        "frozen": 1.5,
        "refrigerated": 1.2,
        "room": 1.0
    }
    shelf_life_factor = shelf_life_factors.get(temperature, 1.0)
    
    return {
        "overall_score": overall_score,
        "temperature_score": temp_score,
        "humidity_score": humidity_score,
        "light_score": light_score,
        "shelf_life_factor": shelf_life_factor
    }

def calculate_nutrition_scores(raw_material_scores, freeze_drying_scores, storage_scores):
    """
    Calculate final nutrition scores based on all factors.
    
    Args:
        raw_material_scores (dict): Raw material evaluation
        freeze_drying_scores (dict): Freeze-drying evaluation
        storage_scores (dict): Storage evaluation
    
    Returns:
        dict: Final nutrition scores
    """
    # Apply nutrient retention factor from freeze-drying
    vitamin_c_base = raw_material_scores["vitamin_c"]
    antioxidant_base = raw_material_scores["antioxidant"]
    
    nutrient_retention = freeze_drying_scores["nutrient_retention_factor"]
    
    # Apply storage degradation
    storage_factor = storage_scores["overall_score"] / 100
    
    # Calculate final scores
    vitamin_c_score = clamp_score(
        vitamin_c_base * nutrient_retention * storage_factor
    )
    antioxidant_score = clamp_score(
        antioxidant_base * nutrient_retention * storage_factor * 0.95  # Antioxidants slightly more stable
    )
    
    return {
        "vitamin_c_score": vitamin_c_score,
        "antioxidant_score": antioxidant_score
    }

def calculate_physical_scores(raw_material_scores, freeze_drying_scores):
    """
    Calculate physical quality scores.
    
    Args:
        raw_material_scores (dict): Raw material evaluation
        freeze_drying_scores (dict): Freeze-drying evaluation
    
    Returns:
        dict: Physical quality scores
    """
    # Color quality based on raw material and freeze-drying
    color_base = raw_material_scores["color"]
    freeze_drying_impact = freeze_drying_scores["overall_score"]
    
    color_quality = calculate_weighted_average([color_base, freeze_drying_impact], [0.6, 0.4])
    
    # Powder flow based on freeze-drying quality
    powder_flow = freeze_drying_scores["overall_score"]
    
    return {
        "color_quality": get_quality_category(color_quality),
        "powder_flow": get_flow_quality(powder_flow)
    }

def calculate_chemical_scores(packaging_scores, storage_scores):
    """
    Calculate chemical stability scores.
    
    Args:
        packaging_scores (dict): Packaging evaluation
        storage_scores (dict): Storage evaluation
    
    Returns:
        dict: Chemical stability scores
    """
    # Moisture stability based on packaging and storage humidity
    protection_score = packaging_scores["protection_score"]
    humidity_score = storage_scores["humidity_score"]
    
    stability_score = calculate_weighted_average([protection_score, humidity_score], [0.6, 0.4])
    
    return {
        "moisture_stability": get_stability_status(stability_score)
    }

def calculate_shelf_life(nutrition_scores, physical_scores, chemical_scores, storage_scores):
    """
    Calculate estimated shelf life.
    
    Args:
        nutrition_scores (dict): Nutrition scores
        physical_scores (dict): Physical scores
        chemical_scores (dict): Chemical scores
        storage_scores (dict): Storage scores
    
    Returns:
        dict: Shelf life estimation
    """
    # Convert categorical scores to numeric for calculation
    physical_numeric = 80  # Default good score
    chemical_numeric = 75  # Default stable score
    
    # Calculate overall nutrition score
    nutrition_overall = calculate_weighted_average(
        [nutrition_scores["vitamin_c_score"], nutrition_scores["antioxidant_score"]], 
        [0.5, 0.5]
    )
    
    # Apply storage shelf life factor
    base_months = estimate_shelf_life_months(nutrition_overall, physical_numeric, chemical_numeric)
    storage_factor = storage_scores["shelf_life_factor"]
    
    estimated_months = int(base_months * storage_factor)
    
    return {
        "estimated_months": estimated_months
    }

def evaluate_jamun_powder_quality(input_data):
    """
    Main evaluation function that processes all input data and returns comprehensive quality assessment.
    
    Args:
        input_data (dict): Complete input data with all sections
    
    Returns:
        dict: Complete quality evaluation results
    """
    # Evaluate each component
    raw_material_scores = evaluate_raw_material(input_data["raw_material"])
    freeze_drying_scores = evaluate_freeze_drying(input_data["freeze_drying"])
    packaging_scores = evaluate_packaging(input_data["packaging"])
    storage_scores = evaluate_storage(input_data["storage"])
    
    # Calculate final scores
    nutrition_scores = calculate_nutrition_scores(
        raw_material_scores, freeze_drying_scores, storage_scores
    )
    physical_scores = calculate_physical_scores(
        raw_material_scores, freeze_drying_scores
    )
    chemical_scores = calculate_chemical_scores(packaging_scores, storage_scores)
    shelf_life_scores = calculate_shelf_life(
        nutrition_scores, physical_scores, chemical_scores, storage_scores
    )
    
    # Combine all results
    return {
        "nutrition": nutrition_scores,
        "physical": physical_scores,
        "chemical": chemical_scores,
        "shelf_life": shelf_life_scores
    }
