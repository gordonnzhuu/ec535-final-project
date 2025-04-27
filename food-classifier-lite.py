import kagglehub

# Download latest version
path = kagglehub.model_download("google/aiy/tfLite/vision-classifier-food-v1")

print("Path to model files:", path)