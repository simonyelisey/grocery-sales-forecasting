DOCKERFILE_NAME = Dockerfile
NAME_IMAGE = sales-forecasting:cp1
NAME_CONTAINER = sales-forecasting

get_data:
	@pip install dvc==3.31.1 && pip install dvc_gdrive==2.20.0
	@dvc fetch data/sells.parquet models/catboost.cbm
	@dvc pull

build:
	@docker build --tag ${NAME_IMAGE} --file ${DOCKERFILE_NAME} .

run:
	@docker run -it --name ${NAME_CONTAINER} ${NAME_IMAGE}

start:
	@docker start ${NAME_CONTAINER}

delete:
	@docker rm ${NAME_CONTAINER}
	@docker rmi ${NAME_IMAGE}
