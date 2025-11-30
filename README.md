# Jamun Powder Quality Evaluator

A Python + Flask rule-based system for evaluating the quality of freeze-dried Jamun powder based on input parameters.

## Features

- **Pure Rule-Based Logic**: No machine learning or database dependencies
- **RESTful API**: Single POST endpoint `/evaluate` for quality assessment
- **Comprehensive Evaluation**: Assesses nutrition, physical, chemical, and shelf life aspects
- **Input Validation**: Thorough validation of all input parameters
- **Production Ready**: Clean, modular, and well-documented code

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoint

#### POST /evaluate

**Request Format:**
```json
{
  "raw_material": {
    "ripeness": "ripe|unripe|overripe",
    "quality_uniformity": true
  },
  "freeze_drying": {
    "temperature": -40,
    "time_hours": 20
  },
  "packaging": {
    "type": "vacuum_pouch|glass_jar|normal_pouch",
    "opaque": true,
    "airtight": true
  },
  "storage": {
    "temperature": "room|refrigerated|frozen",
    "humidity_percent": 5,
    "light_exposure": "dark|light"
  }
}
```

**Response Format:**
```json
{
  "status": "success",
  "results": {
    "nutrition": {
      "vitamin_c_score": 90,
      "antioxidant_score": 85
    },
    "physical": {
      "color_quality": "excellent",
      "powder_flow": "good"
    },
    "chemical": {
      "moisture_stability": "stable"
    },
    "shelf_life": {
      "estimated_months": 8
    }
  }
}
```

### Example Usage with curl

```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "raw_material": {
      "ripeness": "ripe",
      "quality_uniformity": true
    },
    "freeze_drying": {
      "temperature": -40,
      "time_hours": 20
    },
    "packaging": {
      "type": "vacuum_pouch",
      "opaque": true,
      "airtight": true
    },
    "storage": {
      "temperature": "frozen",
      "humidity_percent": 5,
      "light_exposure": "dark"
    }
  }'
```

## Project Structure

```
├── app.py              # Flask API application
├── rules.py            # Rule-based evaluation logic
├── utils.py            # Helper functions and utilities
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Evaluation Rules

### Raw Material Rules
- **Ripe**: High vitamin C score (90)
- **Unripe**: Lower antioxidant score (70)
- **Overripe**: Moderate scores (75-80)
- **Non-uniform quality**: 15% penalty on all scores

### Freeze-Drying Rules
- **Temperature ≤ -40°C**: Best nutrient retention (95%)
- **Time > 30 hours**: 15% nutrient loss
- **Temperature > -20°C**: Poor antioxidant retention (50%)

### Packaging Rules
- **Vacuum + opaque + airtight**: Excellent nutrient protection (+5% bonus)
- **Normal pouch**: 10% oxidation penalty
- **Non-airtight**: 20% oxidation penalty
- **Transparent**: 15% light degradation penalty

### Storage Rules
- **Frozen**: 1.5x shelf life multiplier
- **Humidity > 10%**: Moisture instability
- **Light exposure**: Vitamin C degradation penalty

## Additional Endpoints

- **GET /health**: Service health check
- **GET /**: API documentation and usage examples

## Error Handling

The API provides comprehensive error handling with detailed error messages for:
- Invalid JSON structure
- Missing required fields
- Invalid field values
- Internal server errors

## Development

The codebase is designed to be:
- **Modular**: Clear separation of concerns
- **Maintainable**: Well-commented and documented
- **Extensible**: Easy to add new rules or modify existing ones
- **Testable**: Pure functions that can be easily unit tested
