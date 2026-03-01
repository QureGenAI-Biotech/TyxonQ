import torch
import numpy as np
import os
import argparse
from models.gpt import GPTQE, GPTConfig
from utils.molecule_data import generate_molecule_data
from tyxonq.libs.circuits_library.ucc import build_ucc_circuit

def get_subsequence_energies(op_seq, hamiltonian, hf_state, num_qubits, e_core):
    """Compute energies for sequences of (ex_op, param) operations using TyxonQ.

    Args:
        op_seq: list of lists of (ex_op_tuple, param) pairs
        hamiltonian: scipy sparse matrix (without e_core)
        hf_state: (na, nb) tuple for HF init state
        num_qubits: number of qubits
        e_core: core energy offset
    """
    energies = []
    for ops in op_seq:
        ex_ops_list = [op for op, _ in ops]
        params_list = [float(p) for _, p in ops]
        param_ids = list(range(len(ops)))

        c = build_ucc_circuit(
            params=params_list,
            n_qubits=num_qubits,
            n_elec_s=hf_state,
            ex_ops=ex_ops_list,
            param_ids=param_ids,
            mode='fermion'
        )
        sv = np.asarray(c.state(), dtype=np.complex128).flatten()
        energy = float(np.real(sv.conj() @ hamiltonian @ sv)) + e_core
        energies.append(energy)
    return np.array(energies)

class FeatureMapper:
    """create a mapping between the operation pools of different molecules"""
    def __init__(self, source_ops, target_ops):
        self.source_ops = source_ops
        self.target_ops = target_ops
        self.mapping = self._create_mapping()

    def _create_mapping(self):
        """create a mapping from source operations to target operations
        using string similarity on raw excitation tuples"""
        mapping = {}

        from difflib import SequenceMatcher

        for i, source_op in enumerate(self.source_ops):
            best_match = 0
            best_idx = 0
            source_str = str(source_op)

            for j, target_op in enumerate(self.target_ops):
                target_str = str(target_op)
                similarity = SequenceMatcher(None, source_str, target_str).ratio()

                if similarity > best_match:
                    best_match = similarity
                    best_idx = j

            mapping[i] = best_idx

        return mapping

    def map_indices(self, source_indices):
        """map source indices to target indices"""
        return [self.mapping.get(idx, 0) for idx in source_indices]

class TransferGPTQE(GPTQE):
    """extend GPTQE to support transfer learning"""
    def __init__(self, config, source_vocab_size):
        super().__init__(config)
        self.source_vocab_size = source_vocab_size
        self.config = config

    def adapt_embeddings(self, target_vocab_size):
        """adapt the embedding layer to the new vocabulary size"""
        # get the current device
        device = next(self.parameters()).device

        # save the original weights
        original_wte = self.transformer['wte'].weight.data

        # get the embedding dimension from config
        n_embd = self.config.n_embd

        # create a new embedding layer and move it to the same device
        new_wte = torch.nn.Embedding(target_vocab_size, n_embd).to(device)

        # initialize the new embedding
        if target_vocab_size > self.source_vocab_size:
            # copy the original weights
            new_wte.weight.data[:self.source_vocab_size] = original_wte
            # random initialize the new part
        else:
            # use only part of the original weights
            new_wte.weight.data = original_wte[:target_vocab_size]

        # replace the original embedding layer
        self.transformer['wte'] = new_wte

    def adapt_head(self, target_vocab_size):
        """adapt the model head to the new vocabulary size"""
        # get the current device
        device = next(self.parameters()).device

        # create a new linear layer and move it to the same device
        new_head = torch.nn.Linear(
            self.config.n_embd,  # get the input feature number from config
            target_vocab_size,   # the new vocabulary size
            bias=self.lm_head.bias is not None
        ).to(device)

        # initialize the weights
        if target_vocab_size <= self.source_vocab_size:
            new_head.weight.data = self.lm_head.weight.data[:target_vocab_size]
            if self.lm_head.bias is not None:
                new_head.bias.data = self.lm_head.bias.data[:target_vocab_size]
        else:
            # copy the existing weights, random initialize the new part
            new_head.weight.data[:self.source_vocab_size] = self.lm_head.weight.data
            if self.lm_head.bias is not None:
                new_head.bias.data[:self.source_vocab_size] = self.lm_head.bias.data

        # replace the original layer
        self.lm_head = new_head

def transfer_learning(args):
    """transfer learning from source molecule to target molecule"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. load source molecule data
    source_data = generate_molecule_data(args.source_molecule, use_ucc=True)
    source_mol_data = source_data[args.source_molecule]
    source_op_pool = source_mol_data["op_pool"]
    source_op_pool_size = len(source_op_pool)
    source_ex_ops_raw = source_mol_data["ex_ops_raw"]

    # 2. load target molecule data
    target_data = generate_molecule_data(args.target_molecule, use_ucc=True)
    target_mol_data = target_data[args.target_molecule]
    target_op_pool = target_mol_data["op_pool"]
    target_op_pool_size = len(target_op_pool)
    target_num_qubits = target_mol_data["num_qubits"]
    target_hf_state = target_mol_data["hf_state"]
    target_hamiltonian = target_mol_data["hamiltonian"]
    target_e_core = target_mol_data["e_core"]
    target_ex_ops_raw = target_mol_data["ex_ops_raw"]

    # 3. load the pre-trained model
    source_config = GPTConfig(
        vocab_size=source_op_pool_size + 1,
        block_size=args.seq_len,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=False
    )

    model = TransferGPTQE(source_config, source_op_pool_size + 1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

    # 4. create operation mapping using raw excitation tuples
    feature_mapper = FeatureMapper(source_ex_ops_raw, target_ex_ops_raw)

    # 5. adapt the model to the target molecule
    target_config = GPTConfig(
        vocab_size=target_op_pool_size + 1,
        block_size=args.seq_len,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=False
    )

    # update the model config
    model.config = target_config

    # adapt the embedding layer and output layer
    model.adapt_embeddings(target_op_pool_size + 1)
    model.adapt_head(target_op_pool_size + 1)

    # 6. fine-tune the model (optional)
    if args.fine_tune:
        # freeze most parameters, only fine-tune the key layers
        for name, param in model.named_parameters():
            if 'wte' in name or 'lm_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # generate the fine-tuning data
        train_size = 512  # smaller fine-tuning dataset
        seq_len = args.seq_len
        train_op_pool_inds = np.random.randint(target_op_pool_size, size=(train_size, seq_len))
        train_op_seq = [[target_op_pool[idx] for idx in seq] for seq in train_op_pool_inds]
        train_token_seq = np.concatenate([np.zeros(shape=(train_size, 1), dtype=int), train_op_pool_inds + 1], axis=1)
        train_sub_seq_en = get_subsequence_energies(
            train_op_seq, target_hamiltonian, target_hf_state, target_num_qubits, target_e_core
        )

        tokens = torch.from_numpy(train_token_seq).to(device)
        energies = torch.from_numpy(train_sub_seq_en).to(device)

        # fine-tune
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )

        # fine-tune loop
        model.train()
        for epoch in range(args.fine_tune_epochs):
            epoch_loss = 0
            for token_batch, energy_batch in zip(torch.tensor_split(tokens, 16), torch.tensor_split(energies, 16)):
                optimizer.zero_grad()
                loss = model.calculate_loss(token_batch, energy_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Fine-tuning Epoch {epoch+1}/{args.fine_tune_epochs}, Loss: {epoch_loss:.6f}")

    # 7. predict the target molecule energy
    model.eval()

    with torch.no_grad():
        # ensure the initial index is on the correct device
        idx = torch.zeros(size=(args.n_sequences, 1), dtype=torch.long, device=device)

        # use the same device as the model
        gen_token_seq, pred_Es = model.generate(
            n_sequences=args.n_sequences,
            max_new_tokens=args.seq_len,
            temperature=args.temperature,
            device=device
        )

        # convert to operation sequence
        gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
        gen_inds = np.clip(gen_inds, 0, target_op_pool_size - 1)
        gen_op_seq = [[target_op_pool[idx] for idx in seq] for seq in gen_inds]

        # calculate the true energy
        true_Es = get_subsequence_energies(
            gen_op_seq, target_hamiltonian, target_hf_state, target_num_qubits, target_e_core
        ).reshape(-1, 1)
        pred_Es = pred_Es.cpu().numpy()

        # calculate the error
        mae = np.mean(np.abs(true_Es - pred_Es))
        rmse = np.sqrt(np.mean((true_Es - pred_Es) ** 2))

        print(f"\nResults ({args.source_molecule} -> {args.target_molecule}):")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

        # find the best sequence
        best_idx = np.argmin(true_Es)
        best_energy = true_Es[best_idx].item()
        best_pred_energy = pred_Es[best_idx].item()

        print(f"\nBest energy sequence:")
        print(f"True Energy: {best_energy:.6f}")
        print(f"Predicted Energy: {best_pred_energy:.6f}")
        print(f"Absolute Error: {abs(best_energy - best_pred_energy):.6f}")

        # save the results
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(f'{args.output_dir}/{args.source_molecule}_to_{args.target_molecule}_true_Es.npy', true_Es)
        np.save(f'{args.output_dir}/{args.source_molecule}_to_{args.target_molecule}_pred_Es.npy', pred_Es)

        # save the best model
        if args.save_model:
            torch.save(model.state_dict(),
                       f'{args.output_dir}/{args.source_molecule}_to_{args.target_molecule}_model.pt')

def main():
    parser = argparse.ArgumentParser(description="Transfer learning for molecular energy prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--source_molecule", type=str, default="H2", help="Source molecule name")
    parser.add_argument("--target_molecule", type=str, required=True, help="Target molecule name")
    parser.add_argument("--seq_len", type=int, default=6, help="Sequence length")
    parser.add_argument("--n_sequences", type=int, default=100, help="Number of sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.01, help="Generation temperature")
    parser.add_argument("--fine_tune", action="store_true", help="Whether to fine-tune the model")
    parser.add_argument("--fine_tune_epochs", type=int, default=100, help="Number of fine-tuning epochs")
    parser.add_argument("--output_dir", type=str, default="transfer_results", help="Output directory")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the fine-tuned model")

    args = parser.parse_args()
    transfer_learning(args)

if __name__ == "__main__":
    main()
