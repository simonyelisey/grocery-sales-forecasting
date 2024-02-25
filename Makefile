get_data:
	@pip install dvc && pip install dvc_gdrive
	@dvc pull

build:
	@docker-compose build

up:
	@docker-compose up

run_model:
	@docker-compose exec model python3 main.py
