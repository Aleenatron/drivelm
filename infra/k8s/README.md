# 🚀 Kubernetes Cluster Setup for DriveVLM

This folder contains scripts used to run the DriveVLM Toolkit inside the university's Kubernetes cluster.

---

## 📁 Files

- `vlm4adpvc.yaml`: Defines the persistent volume claim (PVC) for storage  
- `vlm4addep.yaml`: Defines the pod deployment  
- `setup.sh`: Installs Miniconda, sets up Git, and installs essential packages inside the pod

---

## 🧠 Workflow Summary

1. **Launch the persistent volume claim (PVC):**

    ```bash
    kubectl apply -f vlm4adpvc.yaml
    ```

2. **Create the pod and mount the volume:**

    ```bash
    kubectl apply -f vlm4addep.yaml
    ```

3. **Enter the pod after it's running:**

    ```bash
    kubectl exec -it <your-pod-name> -- /bin/bash
    ```

4. **Inside the pod, run the setup script to configure the environment:**

    ```bash
    bash setup.sh
    ```

    This will:
    - Install Miniconda  
    - Set up Git config  
    - Install dependencies (e.g., `ffmpeg`, `tmux`, `unzip`, etc.)  
    - Prepare the Python environment for running the VLM code

---

## 📝 Notes

- Ensure your `.ssh/id_rsa` file is available if required in `setup.sh`
- This setup assumes root access or permissions to install system packages
