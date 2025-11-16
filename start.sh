#!/bin/bash

echo "🚀 Starting Mini Cloud Training Platform..."
echo ""
echo "Building containers..."
docker-compose up --build -d

echo ""
echo "✅ Platform is starting!"
echo ""
echo "Access points:"
echo "  - Frontend: http://localhost:8501"
echo "  - API: http://localhost:8000"
echo "  - MLflow: http://localhost:5000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "Default login: admin / admin123"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
