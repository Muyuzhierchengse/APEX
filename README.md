# APEX

APEX is the research codebase for PolyGNN/PolyGIN and graph-neural-network explanations based on Aumann–Shapley and integral attribution. It has been split from FSX as an independent project; FSX is a separate research work and is not an APEX core method.

## Methods and repository boundary

The active research implementations are PolyGIN, PolyGINExplainer, the algebraic PolyGINExplainer, Integrated Gradients, Gauss–Legendre IG, Trapezoidal IG, Simpson IG, and adaptive Riemann IG. Shared experimental baselines include FlowX, GNNExplainer, GradCAM, PGExplainer, GCN, and GIN.

Traceable variants are retained separately: an Aumann–Shapley candidate, Expected IG, a PolyGNN Shapley prototype, and marginal-OOD evaluation. They have not been merged into a single final algorithm. Legacy material retains the extended data loader, a copied evaluation entry, and an imbalanced-training experiment.

## Repository layout

```text
src/apex/models/       GCN, GIN, and PolyGIN models
src/apex/data/         active dataset loaders
src/apex/explainers/   active explanation methods and shared baselines
src/apex/evaluation/   Fidelity, Stability, and NFE support
src/apex/utils/        reproducibility, checkpoint, and path utilities
scripts/               active training, evaluation, and visualization entries
experiments/variants/  traceable method and evaluation variants
experiments/legacy/    retained historical experiment branches
tests/                 static and CPU lightweight regression tests
data/                  user-provided or runtime-downloaded datasets
artifacts/checkpoints/ model and explainer checkpoints
outputs/logs/          evaluation logs
outputs/results/       training and numerical results
outputs/figures/       generated figures
```

The runtime directories need not exist in a fresh checkout. They are supplied by the user or created by scripts only when those scripts actually run.

## Environment and installation

The lightweight CPU tests have been verified with Python 3.9, PyTorch 2.5.1 CPU, torch-geometric 2.6.1, torch-scatter 2.1.2, torch-sparse 0.6.18, NumPy 1.26.4, and SciPy 1.11.4. The complete research features additionally import DIG (`dive-into-graphs`), Captum, RDKit, NetworkX, Matplotlib, tqdm, OGB, and scikit-learn. Those full dependencies have not all been installed and exercised in the lightweight test environment.

For a Windows CPU environment, install in this order:

1. Install the PyTorch 2.5.1 CPU wheel from the official PyTorch CPU index.
2. Install matching binary wheels for `torch-scatter` and `torch-sparse` from `https://data.pyg.org/whl/torch-2.5.1+cpu.html`; do not fall back to source compilation.
3. Install torch-geometric 2.6.1, NumPy 1.26.4, and SciPy 1.11.4.
4. Install the remaining packages listed in `requirements.txt` when full experiments are required.
5. Install this repository in editable mode for research development:

```powershell
E:\conda_envs\fsx_apex_test\python.exe -I -m pip install --no-deps --no-cache-dir --no-build-isolation -e .
```

For an ordinary, non-editable installation, set the project root before importing the path configuration or running research scripts:

```powershell
$env:APEX_PROJECT_ROOT = "E:\workplace\APEX"
```

`APEX_PROJECT_ROOT` must point to a project root containing both `pyproject.toml` and `src/apex`. An editable/source-checkout installation can discover that verified root. A non-editable installation cannot infer the repository from `site-packages` and raises a clear error when the variable is absent.

## Data and output paths

APEX uses these project-root-relative locations:

- data: `data/`
- checkpoints: `artifacts/checkpoints/`
- evaluation logs: `outputs/logs/`
- numerical results: `outputs/results/`
- figures: `outputs/figures/`

These paths do not depend on the process working directory. Importing the package does not download data or create research results. Scripts create their required output directories only when their entry points run.

## Active capabilities

The active loader supports the BA_shapes node-classification dataset and the BBBP, Graph-SST2, BACE, and Mutagenicity graph-classification datasets. Available models are GCN, GCN_2l, GIN, GIN_2l, and PolyGIN.

The evaluation and experiment entries cover Fidelity+/Fidelity−, Stability/Jaccard, completeness or efficiency gap, NFE, explanation time, exact/algebraic fidelity comparison, and signed-attribution visualization.

## Commands

Run commands from the repository root after installing the package. The examples use current model and dataset choices:

```powershell
E:\conda_envs\fsx_apex_test\python.exe -I scripts/train.py --dataset BBBP --model_used PolyGIN
E:\conda_envs\fsx_apex_test\python.exe -I scripts/evaluate.py --dataset BBBP --model_used PolyGIN --explainer PolyGINExplainer
E:\conda_envs\fsx_apex_test\python.exe -I scripts/evaluate_nfe.py --dataset BBBP --model_used PolyGIN --explainer PolyGINExplainer
E:\conda_envs\fsx_apex_test\python.exe -I scripts/evaluate_fidelity_only.py --dataset BBBP --model_used PolyGIN --explainer PolyGINExplainer
E:\conda_envs\fsx_apex_test\python.exe -I scripts/compare_exact_fidelity.py --dataset BBBP --model_used PolyGIN
E:\conda_envs\fsx_apex_test\python.exe -I scripts/visualize_method_comparison.py
E:\conda_envs\fsx_apex_test\python.exe -I scripts/visualize_signed_attribution.py --device cpu
```

Formal entries may download datasets, and PGExplainer use may train a separate explanation network. No such operation was run during this refactoring.

Run the static and lightweight CPU suites separately:

```powershell
E:\conda_envs\fsx_apex_test\python.exe -I -B -m pytest -p no:cacheprovider -m "not lightweight_graph" --timeout=20 tests
E:\conda_envs\fsx_apex_test\python.exe -I -B -m pytest -p no:cacheprovider -m lightweight_graph --timeout=20 tests
```

The lightweight suite uses tiny artificial graphs; it does not train models or download datasets.

## Known limitations

- The repository currently contains neither APEX checkpoints nor datasets.
- Paper-scale training and end-to-end execution of the formal entries have not been verified on this machine.
- The complete DIG, Captum, RDKit, and OGB environment has not been fully installed and imported in the lightweight test environment.
- `experiments/legacy/train_imbalanced.py` retains its historical Graph-Twitter, extended-data, and `*_3l` contracts; it is not an active entry point.
- Loading an untrusted pickle checkpoint with ordinary `torch.load` can execute malicious code; use only trusted checkpoints.
- Files under `experiments/variants/` are traceable experiment implementations, not a claim that they form one unified final algorithm.
- Citation details still require confirmation from the authors.

## License

This project is distributed under the MIT License. See `LICENSE`.

## Citation

TODO: add the author-confirmed APEX paper title, authors, venue, year, and BibTeX. No citation metadata is inferred here.
