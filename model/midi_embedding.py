import torch
import torch.nn as nn

PAD_ID = 0

class CPWordEmbedding(nn.Module):
    def __init__(self, vocab_size_list, embbed_size_list, d_model=512):
        """
        :param vocab_size_list: List of vocabulary sizes for each token type.
        :param embbed_size_list: List of embedding sizes for each token type.
        :param d_model: Dimension of the model, default is 512.
        """
        super().__init__()
        self.d_model = d_model

        self.family_embedding = nn.Embedding(vocab_size_list[0], embbed_size_list[0], padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(vocab_size_list[1], embbed_size_list[1], padding_idx=PAD_ID)
        self.pitch_embedding = nn.Embedding(vocab_size_list[2], embbed_size_list[2], padding_idx=PAD_ID)
        self.duration_embedding = nn.Embedding(vocab_size_list[3], embbed_size_list[3], padding_idx=PAD_ID)
        self.program_embedding = nn.Embedding(vocab_size_list[4], embbed_size_list[4], padding_idx=PAD_ID)
        self.tempo_embedding = nn.Embedding(vocab_size_list[5], embbed_size_list[5], padding_idx=PAD_ID)
        self.time_signature_embedding = nn.Embedding(vocab_size_list[6], embbed_size_list[6], padding_idx=PAD_ID)

        self.in_linear = nn.Linear(sum(embbed_size_list), self.d_model)

    def forward(self, x):
        """
        Forward pass for the model.
        """
        family_embed = self.family_embedding(x[..., 0])
        position_embed = self.position_embedding(x[..., 1])
        pitch_embed = self.pitch_embedding(x[..., 2])
        duration_embed = self.duration_embedding(x[..., 3])
        program_embed = self.program_embedding(x[..., 4])
        tempo_embed = self.tempo_embedding(x[..., 5])
        time_signature_embed = self.time_signature_embedding(x[..., 6])

        # Concatenate all embeddings
        embeddings = [
            family_embed, position_embed, pitch_embed, duration_embed,
            program_embed, tempo_embed, time_signature_embed
        ]
        # Pass through the input linear layer
        x = self.in_linear(torch.cat(embeddings, dim=-1))
        return x
    
if __name__ == "__main__":
    # example usage
    from datasets import load_from_disk
    import numpy as np

    ds = load_from_disk('../dataset/huggingface_seq2seq_rd')
    test = ds["training"][0]["midi_clean_ids"]
    test = torch.LongTensor(np.array(test))

    vocab_size_list = [3, 131, 130, 102, 131, 34, 66]
    embbed_size_list = [32, 384, 512, 256, 32, 32, 32]
    embedding_layer = CPWordEmbedding(vocab_size_list, embbed_size_list)

    print(embedding_layer.forward(test))