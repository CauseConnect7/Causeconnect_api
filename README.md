# Organization Matching System API

## Overview
This API system provides organization matching functionality using AI embeddings and MongoDB Atlas Vector Search. It supports both Non-Profit and For-Profit organizations, enabling intelligent partnership recommendations.

## Table of Contents
1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [API Endpoints](#api-endpoints)
4. [Installation](#installation)
5. [Environment Setup](#environment-setup)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Testing](#testing)
8. [Database Schema](#database-schema)

## Features
- Organization profile processing
- AI-powered tag generation
- Vector embeddings for similarity matching
- Intelligent partnership recommendations
- Detailed match analysis
- Ad campaign generation
- Health monitoring

## Tech Stack
- **Backend**: FastAPI
- **Database**: MongoDB Atlas
- **AI Services**: OpenAI GPT-4
- **Vector Search**: MongoDB Atlas Vector Search
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## API Endpoints

### Health Check Endpoints
```http
GET /
GET /health
GET /health/database/debug
```

### Organization Processing
```http
POST /test/generate/tags
POST /test/generate/embedding
POST /test/generate/ideal-organizations
POST /test/find-matches
POST /test/evaluate-match
POST /test/complete-matching-process-new
GET /test/company/{company_id}
```

### Ad Campaign Generation
```http
POST /test/analyze/match-reasons
POST /test/generate/ad-campaign
```

## Installation

```bash
# Clone repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --reload
```

## Environment Setup

Required environment variables:
```env
MONGODB_URI=your_mongodb_uri
MONGODB_DB_NAME=your_db_name
MONGODB_COLLECTION_NONPROFIT=nonprofit_collection
MONGODB_COLLECTION_FORPROFIT=forprofit_collection
OPENAI_API_KEY=your_openai_api_key
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: API CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        env:
          MONGODB_URI: ${{ secrets.MONGODB_URI }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          # Add deployment commands here
```

### Deployment Stages
1. **Development**
   - Automatic testing on every push
   - Code quality checks
   - Development environment deployment

2. **Staging**
   - Integration testing
   - Performance testing
   - Staging environment deployment

3. **Production**
   - Final approval check
   - Blue-green deployment
   - Health check verification

## Testing

Run tests using pytest:
```bash
pytest -v
```

Key test areas:
- Profile processing
- Embedding generation
- Match finding
- Ad campaign generation
- Database connections
- API endpoints

## Database Schema

### Organization Collection
```json
{
    "_id": ObjectId,
    "Name": String,
    "Description": String,
    "Type": String,
    "State": String,
    "URL": String,
    "Embedding": Binary,
    "linkedin_description": String,
    "linkedin_industries": String,
    "linkedin_specialities": String,
    "ideal_organization": String
}
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
[MIT License](LICENSE)
