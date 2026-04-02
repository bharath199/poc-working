import kagglehub

# Download latest version
# path = kagglehub.dataset_download("kubeedgeianvs/pcb-aoi")
path = kagglehub.dataset_download("mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection")

print("Path to dataset files:", path)