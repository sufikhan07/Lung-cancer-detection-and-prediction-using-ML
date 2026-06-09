# Lung Cancer Diagnostic System API Documentation

## Overview

The Lung Cancer Diagnostic System provides RESTful APIs for CT scan analysis and risk assessment. All endpoints return JSON responses and use standard HTTP status codes.

**Base URL:** `http://localhost:5000/api/v1`

## Authentication

Currently, the API does not require authentication. In production deployments, consider implementing JWT-based authentication.

## Common Response Format

All API responses follow this structure:

```json
{
  "success": true|false,
  "data": { ... } | null,
  "error": "error message" | null,
  "timestamp": "2024-01-24T10:30:00Z",
  "processing_time": 1.23
}
```

## Endpoints

### Health Check

Get system health status including model availability and system resources.

**Endpoint:** `GET /api/v1/health`

**Response:**
```json
{
  "success": true,
  "overall_status": "healthy",
  "models": {
    "status": "healthy",
    "details": {
      "ct_model_loaded": true,
      "risk_model_loaded": true,
      "scaler_loaded": true,
      "models_functional": true
    }
  },
  "system": {
    "file_system": {
      "status": "healthy",
      "upload_directory_exists": true,
      "upload_directory_writable": true
    }
  },
  "timestamp": "2024-01-24T10:30:00Z"
}
```

### CT Scan Analysis

Analyze uploaded CT scan images for lung cancer detection.

**Endpoint:** `POST /api/v1/ct-scan/analyze`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (file): CT scan image (PNG, JPG, JPEG)

**Success Response:**
```json
{
  "success": true,
  "prediction": "Non-Cancerous",
  "confidence": 87.5,
  "probabilities": {
    "non_cancerous": 87.5,
    "cancerous": 12.5
  },
  "image_path": "/static/uploads/ct_scan_1706092200.jpg",
  "processing_time": 2.34,
  "timestamp": "2024-01-24T10:30:00Z",
  "model_version": "1.0.0"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "No file provided in request",
  "processing_time": 0.01,
  "timestamp": "2024-01-24T10:30:00Z"
}
```

### Risk Assessment

Assess lung cancer risk based on patient factors.

**Endpoint:** `POST /api/v1/risk/assess`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "age": 45,
  "gender": "female",
  "air_pollution": 3,
  "alcohol_use": 2,
  "dust_allergy": 4,
  "occupational_hazards": 3,
  "genetic_risk": 2,
  "chronic_lung_disease": 1,
  "balanced_diet": 7,
  "obesity": 5,
  "smoking": 1,
  "passive_smoker": 2,
  "chest_pain": 3,
  "coughing_blood": 1,
  "fatigue": 4,
  "weight_loss": 2,
  "shortness_of_breath": 3,
  "wheezing": 2,
  "swallowing_difficulty": 1,
  "clubbing": 2,
  "frequent_cold": 4,
  "dry_cough": 3,
  "snoring": 5
}
```

**Success Response:**
```json
{
  "success": true,
  "risk_level": "Medium",
  "risk_percentage": 45.0,
  "confidence": 78.5,
  "probabilities": {
    "Low": 21.5,
    "Medium": 78.5,
    "High": 0.0
  },
  "patient_info": {
    "age": 45,
    "gender": "Female"
  },
  "processing_time": 0.89,
  "timestamp": "2024-01-24T10:30:00Z",
  "model_version": "1.0.0"
}
```

### Model Status

Get status of loaded ML models.

**Endpoint:** `GET /api/v1/models/status`

**Response:**
```json
{
  "models_loaded": true,
  "ct_model_available": true,
  "risk_model_available": true,
  "scaler_available": true,
  "timestamp": 1706092200.123
}
```

### Diagnostics History

Get historical diagnostics data (placeholder for future database integration).

**Endpoint:** `GET /api/v1/diagnostics/history`

**Response:**
```json
{
  "message": "Diagnostics history feature coming soon",
  "total_records": 0,
  "records": [],
  "timestamp": 1706092200.123
}
```

## Risk Factors Scale

All risk factors use a 1-9 scale:

1. **Very Low/Never**
2. **Low/Rarely**
3. **Low-Moderate/Occasionally**
4. **Moderate**
5. **Moderate-High**
6. **High/Frequently**
7. **Very High/Very Frequently**
8. **Extremely High/Always**
9. **Critical/Maximum**

## Error Codes

- `400 Bad Request`: Invalid input data or missing required fields
- `404 Not Found`: Endpoint does not exist
- `405 Method Not Allowed`: HTTP method not supported
- `413 Request Entity Too Large`: File upload exceeds size limit
- `500 Internal Server Error`: Server-side error

## Rate Limiting

API endpoints are rate-limited to prevent abuse:
- 100 requests per minute per IP address
- File uploads limited to 10MB per file

## File Upload Limits

- **Maximum file size:** 10MB
- **Allowed formats:** PNG, JPG, JPEG
- **Image dimensions:** Automatically resized to 224x224 pixels

## Model Information

### CT Scan Model
- **Type:** Convolutional Neural Network (CNN)
- **Architecture:** Custom CNN with ResNet50 backbone
- **Input:** 224x224 RGB images
- **Output:** Binary classification (Cancerous/Non-Cancerous)
- **Accuracy:** ~94%
- **Training data:** 15,000+ annotated CT scans

### Risk Assessment Model
- **Type:** Neural Network
- **Input:** 23 risk factors
- **Output:** Multi-class classification (Low/Medium/High risk)
- **Accuracy:** ~90%
- **Training data:** 1,000+ patient records

## Data Privacy

- Uploaded images are processed in memory and not permanently stored
- Patient data is not stored in the current version
- All processing is done locally on the server
- No data is transmitted to external services

## Versioning

API versioning follows semantic versioning (v1). Breaking changes will result in a new major version.

## Support

For API support or questions:
- Check the health endpoint for system status
- Review error messages for specific issues
- Contact the development team for technical assistance</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\docs\api.md