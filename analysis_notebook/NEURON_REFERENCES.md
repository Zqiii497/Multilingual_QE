# Neuron Identification References

The following repositories provide the reference implementations for the neuron-identification methods used in this project's Analysis notebooks.

## 1. LAPE: Language-Specific Neurons

- Paper: [Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models](https://aclanthology.org/2024.acl-long.309/)
- Official implementation: https://github.com/RUCAIBox/Language-Specific-Neurons
- Relevant analysis: `04_lape_neuron_discovery_recovery.ipynb`

This repository is the official implementation of the ACL 2024 paper and provides the LAPE-based identification code and released language-specific neuron indices. This project uses the paper as the methodological reference; the quantization-oriented recovery experiments and extensions in this repository are project-specific.

## 2. Transfer Neurons

- Paper: [The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs](https://aclanthology.org/2025.emnlp-main.1618/)
- Official implementation: https://github.com/HinaTezuka/emnlp2025-transfer-neurons
- Relevant analysis: `03_transfer_neuron_discovery_recovery.ipynb`

This repository is the authors' implementation accompanying the EMNLP 2025 paper. It is cited as the reference for the Transfer Neurons hypothesis and its neuron-identification approach. The quantization-specific analysis and recovery experiments in this project are extensions and should be attributed to this project.

## Attribution note

These repositories are cited as upstream methodological references, not as the source of this project's experimental results. Please consult the original papers and their repositories for the exact methods, implementation details, and licenses.
