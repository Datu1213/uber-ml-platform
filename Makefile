.PHONY: help install dev-install test lint format clean docker-up docker-down deploy

help:
	@echo "Uber ML Platform - Available Commands"
	@echo "======================================"
	@echo "install        - Install production dependencies"
	@echo "dev-install    - Install development dependencies"
	@echo "test           - Run tests"
	@echo "lint           - Run linters"
	@echo "format         - Format code with black"
	@echo "clean          - Clean build artifacts"
	@echo "docker-up      - Start Docker services"
	@echo "docker-down    - Stop Docker services"
	@echo "deploy-local   - Deploy to local Kubernetes"
	@echo "deploy-prod    - Deploy to production"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .
	isort .

type-check:
	mypy .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

docker-up:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Services started. Access at:"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Airflow: http://localhost:8080"
	@echo "  - Grafana: http://localhost:3000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

feast-apply:
	cd feature_store && feast apply

feast-materialize:
	cd feature_store && feast materialize-incremental $(shell date -u +%Y-%m-%dT%H:%M:%S)

train-eta:
	python training/train_eta_model.py

serve:
	python serving/main.py

deploy-local:
	kubectl apply -f infrastructure/k8s/deployments.yaml
	kubectl get pods -n ml-platform

deploy-prod:
	cd infrastructure/terraform && terraform apply -auto-approve
	kubectl apply -f infrastructure/k8s/deployments.yaml

k8s-status:
	kubectl get all -n ml-platform

k8s-logs:
	kubectl logs -f -n ml-platform -l app=inference-service

monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	kubectl port-forward -n ml-platform svc/grafana-service 3000:3000 &
	kubectl port-forward -n ml-platform svc/prometheus-service 9090:9090 &

init-project:
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	make docker-up
	make feast-apply
	@echo "Project initialized successfully!"

.DEFAULT_GOAL := help
