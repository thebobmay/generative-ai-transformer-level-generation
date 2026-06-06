# Conditional Transformer-Based Generation of 2D Platformer Level Segments

**Student:** Robert Mayfield

---

## Project Description

This project trains a conditional decoder only Transformer to generate 2D platformer level segments from tile based level data. The model learns from Super Mario Bros and Super Mario Bros: The Lost Levels levels in the Video Game Level Corpus (VGLC) and uses prepended difficulty control tokens to condition generated output on Easy, Medium, or Hard structural profiles. Difficulty labels are derived from heuristic structural features: enemy density, longest ground gap, hazard density, solid tile ratio, and collectible density. The project trains both a conditional model and a baseline unconditional model, compares their outputs, evaluates four generation metrics (structural validity, diversity, nearest-neighbour similarity, and difficulty match), and runs corpus expansion and sampling strategy experiments to characterize the limits of the approach.

---

## Dataset

**Name:** Video Game Level Corpus (VGLC)
**Source:** Summerville, A., Snodgrass, S., Mateas, M., & Ontañón, S. (2016). The VGLC: The video game level corpus. *Proceedings of the 7th Workshop on Procedural Content Generation.*
**Repository:** [https://github.com/TheVGLC/TheVGLC](https://github.com/TheVGLC/TheVGLC)
**License:** CC BY 4.0

**Levels used:**

- Super Mario Bros overworld levels: 15 levels (mario-1-1 through mario-8-1)
- Super Mario Bros: The Lost Levels: 20 levels (22 downloaded, 2 skipped due to non-normalizable row heights)

All level files are committed directly to `data/raw/`. No external download is required.

---

## Files Included

```
notebooks/generative_model.ipynb               main project notebook
reports/Generative_AI_Analysis_Report.pdf      full written report
environment.yml                                conda environment (authoritative GPU spec)
requirements.txt                               pip package dependencies
data/raw/                                      VGLC level .txt files and smb.json tile spec
outputs/figures/                               training curves, difficulty distributions,
                                               confusion matrices, sampling experiment chart
outputs/tables/evaluation_metrics.csv          baseline evaluation metrics
outputs/tables/evaluation_metrics_expanded.csv baseline vs expanded corpus comparison
outputs/tables/sampling_experiment_results.csv temperature, top-k, and top-p results
outputs/model_artifacts/level_generator.pt     conditional model weights (difficulty-conditioned)
outputs/model_artifacts/baseline_generator.pt  baseline unconditional model weights
outputs/model_artifacts/level_generator_expanded.pt  expanded corpus conditional model weights
outputs/model_artifacts/level_tokenizer.pkl    vocabulary and encoding wrapper
outputs/model_artifacts/difficulty_scorer.pkl  heuristic scorer with fitted thresholds
src/data_processing.py                         level loading, normalization, and chunking
src/feature_engineering.py                     structural feature computation
src/labeling.py                                difficulty scoring and label assignment
src/tokenizer.py                               vocabulary, tokenization, and Dataset class
src/model.py                                   LevelTransformer architecture
src/artifacts.py                               LevelTokenizer and DifficultyScorer (pickled classes)
src/evaluation.py                              generation metric functions
```

`level_generator.pt` and `evaluation_metrics.csv` are the primary submission artifacts. The `_expanded` variants (`level_generator_expanded.pt`, `evaluation_metrics_expanded.csv`) document the follow-up corpus expansion experiment, and `baseline_generator.pt` is the unconditional reference model from the original analysis.

---

## How to Run

1. Create the conda environment from `environment.yml`:
  ```
   conda env create -f environment.yml
  ```
2. Activate the environment:
  ```
   conda activate generative-ai-capstone
  ```
3. Open the notebook in Jupyter:
  ```
   jupyter notebook notebooks/generative_model.ipynb
  ```
4. Run all cells from top to bottom.

**Prerequisites:** Anaconda or Miniconda with Python 3.11 and conda on your PATH.

**Note:** GPU execution is strongly recommended. The notebook detects CUDA automatically and falls back to CPU if unavailable. Training both models for 150 epochs takes under 5 minutes on an RTX 3080. The expanded corpus experiment trains a third model under the same hyperparameters. The sampling experiment generates approximately 1,800 level samples and may take several minutes on CPU.

The `environment.yml` pins `torch==2.6.0+cu124` (CUDA 12.4 build). CUDA-variant torch builds are not hosted on PyPI, so the pip section of `environment.yml` includes `--extra-index-url https://download.pytorch.org/whl/cu124` to resolve them. If your solver does not honor that line, install torch explicitly after activating the environment:

```
pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

CPU-only users should instead install the CPU build:

```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Note on requirements.txt:** This file was generated with `pip freeze` from the `generative-ai-capstone` conda environment. Many entries use the `package @ file:///...` format, which is standard output for conda-managed packages when captured via pip. To reproduce the environment, use `conda env create -f environment.yml` rather than `pip install -r requirements.txt`.

---

## Bias and Responsible Data Handling

The training corpus is drawn entirely from two Nintendo titles released in 1985 and 1986, both built on the same game engine with the same tile vocabulary and level design conventions. Generated outputs reflect the structural patterns of this specific corpus and should not be treated as representative of platformer level design in general. Levels from other genres, game engines, or design eras are not represented.

The heuristic difficulty labels are derived from structural features of 32 column tile chunks and are not validated against player playtesting data, completion rates, or accessibility assessments. A chunk labeled Hard by this system may not be challenging for an experienced player, and a chunk labeled Easy may be impassable for a new player. Any downstream use of generated levels should involve human review before treating the difficulty labels as ground truth.

The nearest neighbour similarity analysis confirmed that a substantial fraction of generated outputs are near-verbatim reproductions of training data. Systems that deploy this model in a context where content originality is assumed should disclose this limitation and apply appropriate filtering.

---

## Future Integration Reflection

### How this generator could support future projects

This level generator can serve as the procedural content layer in a larger system: given a target difficulty level for the current player session, the generator produces candidate level segments that match the requested structural profile. A higher level controller or orchestrating agent can select among generated candidates based on additional context such as recent player performance, session length, or inferred skill level. The saved `level_tokenizer.pkl` and `difficulty_scorer.pkl` artifacts provide a consistent interface for loading the generation and scoring components without re-running training, so a future project can reuse them directly.

### How this dataset and model would need to evolve for deeper integration

Deeper integration would require several expansions. The training corpus currently covers only two Mario titles and 35 levels. Adding data from other VGLC platformers, custom designed levels, or procedurally verified levels would broaden the structural vocabulary and reduce the memorization behavior observed in evaluation. The difficulty scoring system would benefit from player validated labels rather than heuristic proxies, and the scoring weights would need recalibration for non-Mario tile vocabularies. A masked loss training approach that redirects gradient signal away from the dominant sky tile and toward structural tiles is a candidate near term improvement that requires no architectural changes.

### How agentic automation could assist this workflow

An agentic pipeline could automate the end to end level generation loop: receiving a difficulty target from a higher level controller, sampling a candidate level at the recommended temperature setting (1.2 for balanced diversity and accuracy), scoring it with the difficulty scorer, and returning only candidates that pass both the structural validity check and a difficulty threshold. The pipeline could also run batch generation to build a candidate library ahead of time, indexed by difficulty class, so the controller can retrieve a pre-validated segment without incurring generation latency during an active session.

---

## Requirements

See `environment.yml` for the full conda environment including CUDA dependencies.
See `requirements.txt` for pip package dependencies.

Key libraries: Python 3.11, PyTorch 2.6.0 (CUDA 12.4), NumPy 2.4.3, pandas 3.0.1, scikit-learn 1.7.1, matplotlib 3.10.8, seaborn 0.13.2, JupyterLab 4.5.3