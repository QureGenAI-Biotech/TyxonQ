# how to use the code

## 1.train
- Change the molecule in `molecule_data = generate_molecule_data("H2", use_ucc=True)` in `training/train_gptqe.py`.

- Molecules can be from https://pennylane.ai/datasets/collection/qchem

- run `python main.py`

- ckpt files will be saved at `checkpoints/*`


## 2.predict
- `python predict_gptqe.py --model_path ./checkpoints/model.pt --molecule H2`


## 3.transfer

- without fine-tune: `python transfer_gptqe.py --model_path ./checkpoints/model.pt --source_molecule H2 --target_molecule H4 --n_sequences 200`

- with fine-tune: `python transfer_gptqe.py --model_path ./checkpoints/model.pt --source_molecule H2 --target_molecule H4 --n_sequences 200 --fine_tune --fine_tune_epochs 200 --save_model`