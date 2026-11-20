# Linux/macOS convenience targets. On Windows use:  ./tasks.ps1 <target>
PY ?= .venv/bin/python
.PHONY: setup scrape normalize index run test lint eval docker docker-run deploy

setup:
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt -r requirements-scrape.txt -r requirements-dev.txt
	$(PY) -m playwright install chrome || true
scrape:
	$(PY) -m scraper.wasalt --sample 4000
	$(PY) -m scraper.darglobal
normalize:
	$(PY) -m scraper.normalize
index:
	$(PY) -m app.build_index --write-cache
run:
	$(PY) -m uvicorn app.main:app --port 7860 --reload
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
eval:
	$(PY) eval/run_eval.py --base-url http://localhost:7860
docker:
	docker build -t realestate-chatbot .
docker-run:
	docker run --rm -p 7860:7860 --env-file .env realestate-chatbot
deploy:
	$(PY) scripts/deploy_hf.py
