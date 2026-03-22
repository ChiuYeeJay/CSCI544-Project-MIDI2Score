from datasets import load_from_disk

ds = load_from_disk('dataset/huggingface')
print(f"training set size: {len(ds['training'])}")
# print(f"validation set size: {len(ds['validation'])}")
# print(f"test set size: {len(ds['test'])}")

train_data = ds["training"].with_format("torch")
for data in train_data:
    print(data["input_ids"])
    break
