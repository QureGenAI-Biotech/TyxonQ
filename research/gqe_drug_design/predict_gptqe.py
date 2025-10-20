import torch
import numpy as np
import argparse
import os
from models.gpt import GPTQE, GPTConfig
from utils.molecule_data import generate_molecule_data
import pennylane as qml

def get_subsequence_energies(op_seq, hamiltonian, init_state, num_qubits):
    
    """Computes the energies for a sequence of operations."""
    dev = qml.device("default.qubit", wires=num_qubits, shots=1024)

    @qml.qnode(dev)
    def energy_circuit(gqe_ops):
        qml.BasisState(init_state, wires=range(num_qubits))
        for op in gqe_ops:
            qml.apply(op)
        return qml.expval(hamiltonian)
    
    
    energies = []
    for ops in op_seq:
        es = energy_circuit(ops)
        energies.append(es.item())
    return np.array(energies)

def load_model(model_path, config):
    """Load a trained GPTQE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTQE(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_energy(model, molecule_name, seq_len=6, n_sequences=100, temperature=0.01):
    """Predict energy for a given molecule using the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate molecule data
    molecule_data = generate_molecule_data(molecule_name, use_ucc=True, use_lie_algebra=True)
    mol_data = molecule_data[molecule_name]
    op_pool = mol_data["op_pool"]
    num_qubits = mol_data["num_qubits"]
    init_state = mol_data["hf_state"]
    hamiltonian = mol_data["hamiltonian"]
    
    op_pool_size = len(op_pool)
    
    # Generate predictions
    with torch.no_grad():
        gen_token_seq, pred_Es = model.generate(
            n_sequences=n_sequences,
            max_new_tokens=seq_len,
            temperature=temperature,
            device=device
        )
        
    # Convert predicted energies to numpy array
    pred_Es = pred_Es.cpu().numpy()
    
    # Calculate true energies for comparison
    gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
    gen_inds = np.clip(gen_inds, 0, op_pool_size - 1)
    gen_op_seq = [[op_pool[idx] for idx in seq] for seq in gen_inds]
    true_Es = get_subsequence_energies(gen_op_seq, hamiltonian, init_state, num_qubits).reshape(-1, 1)
    
    return gen_op_seq, pred_Es, true_Es

def main():
    parser = argparse.ArgumentParser(description="使用预训练的GPTQE模型预测分子能量")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型检查点路径")
    parser.add_argument("--molecule", type=str, default="H2", help="要预测的分子名称 (默认: H2)")
    parser.add_argument("--seq_len", type=int, default=6, help="序列长度 (默认: 6)")
    parser.add_argument("--n_sequences", type=int, default=100, help="要生成的序列数量 (默认: 100)")
    parser.add_argument("--temperature", type=float, default=0.01, help="生成温度 (默认: 0.01)")
    parser.add_argument("--output_dir", type=str, default="predictions", help="保存预测结果的目录 (默认: predictions)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model configuration from the training parameters
    # This should match the configuration used during training
    config = GPTConfig(
        vocab_size=None,  # Will be updated based on molecule data
        block_size=args.seq_len,
        n_layer=12, 
        n_head=12,
        n_embd=768,
        dropout=0.0,  # Set to 0 for inference
        bias=False
    )
    
    # Generate molecule data to get the operator pool size
    molecule_data = generate_molecule_data(args.molecule, use_ucc=True, use_lie_algebra=True)
    mol_data = molecule_data[args.molecule]
    op_pool_size = len(mol_data["op_pool"])
    
    # Update vocab size
    config.vocab_size = op_pool_size + 1
    
    # Load model
    model = load_model(args.model_path, config)
    
    # Predict energies
    gen_op_seq, pred_Es, true_Es = predict_energy(
        model, 
        args.molecule, 
        seq_len=args.seq_len, 
        n_sequences=args.n_sequences,
        temperature=args.temperature
    )
    
    # Calculate statistics
    mae = np.mean(np.abs(true_Es - pred_Es))
    rmse = np.sqrt(np.mean((true_Es - pred_Es) ** 2))
    
    # Print results
    print(f"Predictions for molecule: {args.molecule}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    
    # Find the best sequence (lowest true energy)
    best_idx = np.argmin(true_Es)
    best_energy = true_Es[best_idx].item()
    best_pred_energy = pred_Es[best_idx].item()
    best_sequence = gen_op_seq[best_idx]
    
    print(f"\nBest energy sequence found:")
    print(f"True Energy: {best_energy:.6f}")
    print(f"Predicted Energy: {best_pred_energy:.6f}")
    print(f"Absolute Error: {abs(best_energy - best_pred_energy):.6f}")
    
    # Save results
    np.save(os.path.join(args.output_dir, f"{args.molecule}_true_energies.npy"), true_Es)
    np.save(os.path.join(args.output_dir, f"{args.molecule}_pred_energies.npy"), pred_Es)
    
    # Save detailed results to CSV
    import pandas as pd
    results = pd.DataFrame({
        'True_Energy': true_Es.flatten(),
        'Predicted_Energy': pred_Es.flatten(),
        'Absolute_Error': np.abs(true_Es.flatten() - pred_Es.flatten())
    })
    results.to_csv(os.path.join(args.output_dir, f"{args.molecule}_predictions.csv"), index=False)
    
    # Save best sequence details
    with open(os.path.join(args.output_dir, f"{args.molecule}_best_sequence.txt"), 'w') as f:
        f.write(f"Molecule: {args.molecule}\n")
        f.write(f"True Energy: {best_energy:.6f}\n")
        f.write(f"Predicted Energy: {best_pred_energy:.6f}\n")
        f.write(f"Absolute Error: {abs(best_energy - best_pred_energy):.6f}\n\n")
        f.write("Best Sequence:\n")
        for i, op in enumerate(best_sequence):
            f.write(f"Step {i+1}: {op}\n")

if __name__ == "__main__":
    main()


#python predict_gptqe.py 
# --model_path ./checkpoints/seq_len=6/gptqe_best_model.pt 
# --molecule H2 
# --n_sequences 200 (要生成的序列数量)
# --output_dir predictions