docker run --name bayesianrl-sumo -it --rm \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    --env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --user=$USER \
    -v $(pwd):/app \
    bayesianrl-sumo \
    bash