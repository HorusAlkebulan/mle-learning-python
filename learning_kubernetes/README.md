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

