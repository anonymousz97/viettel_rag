
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW

# Load the T5 model
tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")
model = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base")

# Modify the last layer of the decoder part
config = model.config
config.num_labels = 2
model.resize_token_embeddings(len(tokenizer))
model.lm_head = torch.nn.Linear(config.d_model, config.num_labels)

print(model)

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer

        with open(file_path, 'r') as file:
            for line in file:
                question, answer, groundtruth = line.strip().split('\t')
                input_text = f"question: {question} context: {answer}"
                input_ids = self.tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=model.max_length)
                label = int(groundtruth)
                self.data.append((input_ids, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, label = self.data[index]
        input_ids = torch.tensor(input_ids)
        label = torch.tensor(label)
        return input_ids, label

# Set the file path and model name
file_path = '/path/to/your/file.txt'


# Create the dataset and data loader
dataset = CustomDataset(file_path, tokenizer)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    total_loss = 0
    for batch in data_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, labels=input_ids)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, 2), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Average Loss = {total_loss / len(data_loader)}")

# Save the modified model
model.save_pretrained('/path/to/save/model')
