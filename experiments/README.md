
## Installation
- git clone
- cd nmtscore && pip install ".[prism]"
- Install fork of SacreRouge
  - git clone https://github.com/jvamvas/sacrerouge.git
  - pip install -r sacrerouge/requirements.txt
  - mv sacrerouge/sacrerouge sacrerouge_
  - rm -r sacrerouge
  - mv sacrerouge_ sacrerouge
- cd experiments && pip install -r requirements.txt

## Paraphrase Identification

### Prepare data
- python [scripts/download_data.py](scripts/download_data.py)

### Main experiments (Table 1)
- python [scripts/run_paraphrase_identification.py](scripts/run_paraphrase_identification.py)

### Cross-lingual PAWS-X (Table 2)
- python [scripts/run_crosslingual_pawsx.py](scripts/run_crosslingual_pawsx.py)

### Correlation analysis (Table 3)
- python [scripts/run_correlation_analysis.py](scripts/run_correlation_analysis.py)

### Normalization ablation (Table 4)
- python [scripts/run_normalization_ablation.py](scripts/run_normalization_ablation.py)

### Inference time (Appendix A)
- python [scripts/inference_time.py](scripts/inference_time.py)

### M2M100 models (Appendix E)
- python [scripts/run_m2m100.py](scripts/run_m2m100.py)
- python [scripts/run_m2m100_crosslingual.py](scripts/run_m2m100_crosslingual.py)

## Data-to-text

### RDF-to-text (Figure 2)
- python [scripts/run_rdf_to_text.py](scripts/run_rdf_to_text.py) <en/ru> <system_level/global>

### AMR-to-text (Figure 3)
- python [scripts/run_amr_to_text.py](scripts/run_amr_to_text.py)
