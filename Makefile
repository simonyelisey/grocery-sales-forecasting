DOCKERFILE_NAME = Dockerfile
NAME_IMAGE = sales-forecasting:cp1

build:
	@docker build --tag ${NAME_IMAGE} --file ${DOCKERFILE_NAME} .

save_img:
	@docker save ${NAME_IMAGE} | gzip > ${NAME_IMAGE}.tar.gz
	@du -sh ${NAME_IMAGE}.tar.gz
