# QCNN Multi-Worker Training on Kubernetes

## Setup

If at any point in the setup below you run into permission issues, check that your user account is assigned a role that includes permissions from the following roles:
* container.admin
* iam.serviceAccountAdmin
* storage.admin

To check your permissions, run:

```
gcloud projects get-iam-policy ${PROJECT}
```

and look for your user account. A list of roles can be found [here](https://cloud.google.com/iam/docs/understanding-roles).

---

We've provided a script, `setup.sh`, which runs the setup steps below and should work for most Google Cloud setups. Edit the configuration section in the script, then run `./setup.sh infra` to provision Google Cloud resources, and `./setup.sh param` to fill in parameters in your multi-worker job setup.

This section walks through each step in detail as reference. Please feel free to skip this if `setup.sh` works out of the box.

---

The following variables are used in commands below:
* `${CLUSTER_NAME}`: your Kubernetes cluster name on Google Kubernetes Engine.
* `${PROJECT}`: your Google Cloud project ID.
* `${NUM_NODES}`: the number of VMs in your cluster.
* `${MACHINE_TYPE}`: the [machine type](https://cloud.google.com/compute/docs/machine-types) of VMs. This controls the amount of CPU and memory resources for each VM.
* `${SERVICE_ACCOUNT_NAME}`: The name of both the Google Cloud IAM service account and the associated Kubernetes service account.
* `${ZONE}`: Google Cloud zone for the Kubernetes cluster.
* `${BUCKET_REGION}`: Google Cloud region for Google Cloud Storage bucket. This is recommended to be the region containing your cluster’s zone. The region of a zone is the part of the zone name without the section after the last hyphen. For example, the region of zone "us-west1-a" is "us-west1".
* `${BUCKET_NAME}`: Name of the Google Cloud Storage bucket for storing training output. The name must satisfy [Bucket naming requirements](https://cloud.google.com/storage/docs/naming-buckets#requirements). Example: `your-project-id-qcnn-multinode`.

---

* Set parameters in Kubernetes YAML files
  * Look for parameters surrounded by `<>` in
    * `Makefile`
    * `common/sa.yaml`
    * `training/qcnn.yaml`
    * `inference/qcnn_inference.yaml`
* Set up Google Container Registry: https://cloud.google.com/container-registry/docs/quickstart
  * This is for storing Docker images. Other non-Google container registries recognized by Docker works as well.
* Google Kubernetes Engine (GKE) setup: follow the quick start guide and stop before “Creating a GKE Cluster”: https://cloud.google.com/kubernetes-engine/docs/quickstart#local-shell
* Create a GKE cluster
  * `gcloud container clusters create ${CLUSTER_NAME} --workload-pool=${PROJECT}.svc.id.goog --num-nodes=${NUM_NODES} --machine-type=${MACHINE_TYPE} --zone=${ZONE} --preemptible`
* Workload identity IAM commands:
  * This feature enables the binding between Kubernetes service accounts and Google Cloud service accounts.
  * `gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME}`
  * `gcloud iam service-accounts add-iam-policy-binding   --role roles/iam.workloadIdentityUser   --member "serviceAccount:${PROJECT}.svc.id.goog[default/${SERVICE_ACCOUNT_NAME}]"   ${SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com`
* Google Cloud Storage commands
  * This is for storing training output.
  * Install gsutil: `gcloud components install gsutil`
    * This is pre-installed if you opted to use Cloud Shell.
  * Create bucket: `gsutil mb -p ${PROJECT} -l ${BUCKET_REGION} -b on gs://${BUCKET_NAME}`
* Give service account Cloud Storage permissions: `gsutil iam ch serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT}.iam.gserviceaccount.com:roles/storage.admin gs://${BUCKET_NAME}`
* Install `tf-operator` from Kubeflow: `kubectl apply -f https://raw.githubusercontent.com/kubeflow/tf-operator/v1.0.1-rc.1/deploy/v1/tf-operator.yaml`

### Billable Resources
* Container Registry ([pricing](https://cloud.google.com/container-registry/pricing))
* Kubernetes Engine ([pricing](https://cloud.google.com/kubernetes-engine/pricing))
  * Kubernetes Engine follows Compute Engine [pricing](https://cloud.google.com/compute/all-pricing) for VMs in the cluster. Specifically, the following pricing is relevant for this tutorial:
    * [VM instance pricing](https://cloud.google.com/compute/vm-instance-pricing)
    * [Network pricing](https://cloud.google.com/vpc/network-pricing)
      * The TensorBoard Service contains a [load balancer](https://cloud.google.com/vpc/network-pricing#lb) to serve external traffic.
    * [Disk and image pricing](https://cloud.google.com/compute/disks-image-pricing)
* Cloud Storage ([pricing](https://cloud.google.com/storage/pricing))

Prices can be estimated using the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator)


## Run Training

* `make training`
  * This builds the Docker image, uploads it to the Google Container Registry, and deploys the setup to your GKE cluster.
* (optional) Check TFJob deployment status: `kubectl describe tfjob qcnn`
  * Within the `Status` section, it should eventually say `TFJobCreated` and `TFJobRunning`
* Check worker and Tensorboard pod status: `kubectl get pods`
  * There should be 3 containers: `qcnn-tensorboard-<some_suffix>`, `qcnn-worker-0`, and `qcnn-worker-1`.
  * They should all eventually be in `Running` status. If the status is either `ContainerCreating` or `CrashloopBackoff`, something is wrong.
* Check worker logs: `kubectl logs -f qcnn-worker-0`.
  * Ctrl-C to terminate the log stream.
  * Logs should show progress bars for training epochs, profiler start & end, and eventually writing model weights to a file at the end.
* Set up Tensorboard
  * `make tensorboard`
  * Get the IP of the Tensorboard instance
    * `kubectl get svc tensorboard-service`
    * The IP is under `EXTERNAL-IP`.
    * If the IP is `<pending>`, the load balancer is still being provisioned. Watch the status by running `kubectl get svc tensorboard-service -w`. Eventually the IP should show up.
  * In a browser, go to `<ip>:5001` to access the Tensorboard UI.
  * Due to the lack of support for multiple ports in tf-operator currently ([feature request](https://github.com/kubeflow/tf-operator/issues/1251)), in order to enable profiler sampling mode (for multi-worker profiling), worker Services need to be patched manually with the required profiler port. To do this, run `training/apply_profiler_ports.sh`.
* Training data and log summary data (used by Tensorboard) can be viewed in the [Google Cloud Storage browser](https://console.cloud.google.com/storage/browser)

## Run Inference

* `make inference`
  * This builds the Docker image, uploads it to the Google Container Registry, and deploys the inference job to your GKE cluster.
* Check inference pod status: `kubectl get pods`
  * There should be 1 container: `inference-qcnn-<some_suffix>`.
* Check inference logs to verify inference results: `kubectl logs -f inference-qcnn-<some_suffix>`.

## Cleanup

Cleanup can be done on an as-needed basis.

### Cleaning up billable resources
* `make delete-training` and `make delete-inference`.
  * Removes Kubernetes deployments.
* [Delete Container Registry images](https://cloud.google.com/container-registry/docs/managing#deleting_images)
* [Delete Cloud Storage data](https://cloud.google.com/storage/docs/deleting-objects)
* Delete GKE cluster: `gcloud container clusters delete ${CLUSTER_NAME}`


### Other Cleanup
* Delete docker images: `make remove-images`
  * The next new build of container images will be slow.
* [Delete the Google Cloud service account](https://cloud.google.com/iam/docs/creating-managing-service-accounts#deleting).
