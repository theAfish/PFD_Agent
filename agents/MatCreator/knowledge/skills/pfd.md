---
name: pfd
description: Iterative and robust pre-training, fine-tuning and distillation workflow for ML force fields from pre-trained models.
tags: [iterative, active_learning, model_training]
allowed_agents: [structure_agent, abacus_agent, dpa_agent, plot_agent]
triggers: [fine-tune, finetune, distill, distillation, active learning, convergence, pre-trained]
---
PFD workflows coordinate iterative fine-tuning or distillation of ML force fields from a pre-trained model.

Standard loop:
1. Structure building
2. MD exploration
3. Entropy-based data curation
4. Labeling (DFT or DPA model)
5. Training
6. Convergence check
7. Repeat until criteria or max iterations.

Workflow variants:
- Fine-tuning: ABACUS DFT labels, then continue training existing model on all collected data.
- Distillation: DPA teacher labels, then train a new student model from scratch.

Key parameters to confirm:
- task type (fine-tune/distill), max iterations, convergence criterion
- initial structure, supercell size, perturbation settings
- MD ensemble/temperature/time/steps/save interval
- curation max_sel/chunk_size
- labeling setup (ABACUS kspacing or DPA head)
- training epochs and split
