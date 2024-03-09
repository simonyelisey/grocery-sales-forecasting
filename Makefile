get_data:
	@pip install dvc && pip install dvc_gdrive
	@dvc pull

build:
	@docker-compose build

up:
	@docker-compose up

migrate:
	@docker-compose exec web python manage.py migrate --noinput
