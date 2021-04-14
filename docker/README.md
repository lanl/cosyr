### Build image

```bash
docker images
docker system prune
docker build --build-arg SSH_KEY="${MY_KEY}" \
             --build-arg http_proxy="${HTTP_PROXY}" \
             --build-arg https_proxy="${HTTPS_PROXY}" \
             -t cosyr docker
docker tag cosyr:latest hobywan/cosyr:latest
docker push hobywan/cosyr:latest
```