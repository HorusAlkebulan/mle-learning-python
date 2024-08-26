# Learning Kubernetes

SOURCE: <https://www.linkedin.com/learning/learning-kubernetes-16086900/what-is-kubernetes?autoSkip=true&contextUrn=urn%3Ali%3AlearningCollection%3A7230239471168843778&resume=false&u=0>

## Setup

To install the latest minikube stable release on ARM64 macOS using Homebrew:

If the Homebrew Package Manager is installed:

```sh
brew update
brew install minikube
```

If which minikube fails after installation via brew, you may have to remove the old minikube links and link the newly installed binary:

```sh
brew unlink minikube
brew link minikube
```

## Getting Started

Common commands

```sh
minikube start
kubectl cluster-info
kubectl get nodes
kubectl get namespaces
kubectl get pods -A
kubectl get services -A
```

### Using Minikube Tunnel for Load Balancer Testing

```sh
minikube tunnel
```

## Basic Operations

```sh
kubectl get pods -n development -o wide
kubectl tunnel
minikube tunnel

kubectl apply -f busybox.yaml
kubectl get pods
kubectl exec -it busybox-647b869f4d-s54zl -- /bin/sh
kubectl get pods -A
kubectl logs pod-info-deployment-5cdffc94c-jc86c -n development
kubectl apply -f service.yaml
kubectl get services -n development

kubectl delete -f busybox.yaml
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
kubectl delete -f namespace.yaml
minikube delete
```
