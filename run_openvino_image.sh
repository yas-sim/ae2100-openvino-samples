#!/usr/bin/env bash

docker run --device /dev/dri \
	-it \
	--device=/dev/ion:/dev/ion \
	-v /var/tmp:/var/tmp \
	-v /home/root/tmp:/root/tmp \
	--name ubuntu-openvino \
	-d ubuntu:openvino_2020R3 \
	/bin/bash
