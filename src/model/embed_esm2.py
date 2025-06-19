import torch
import esm

def get_esm2_embeddings(sequences):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_representations = results["representations"][33]

    embeddings = []
    for i, (_, seq) in enumerate(data):
        seq_len = len(seq)
        emb = token_representations[i, 1:seq_len + 1].cpu()
        embeddings.append(emb)

    return embeddings

if __name__ == "__main__":
    protein_x = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    embs = get_esm2_embeddings([protein_x])
    emb = embs[0]  # Get the embedding tensor for this protein

    print(f"Embedding shape for protein_x: {emb.shape}")

    # Print the whole matrix (all residues' embeddings)
    print("Full embedding matrix:")
    print(emb)

    # Print the embedding vector for the first amino acid
    print("\nFirst amino acid embedding vector:")
    print(emb[0])

    # Print the embedding vector as a list
    print("\nFirst amino acid vector as list of floats:")
    print(emb[0].tolist())                                                                                                                               