# Ensemble_ML for MetaTrader 5

This repository contains the MetaTrader 5 Expert Advisors and runtime assets used for the Ensemble ML forex-trading experiments.

## Contents

- `DataExporter.mq5`
  Exports H1 OHLCV data and indicator-based features from MT5 for offline model training.

- `EnsembleOnnxTrader.mq5`
  Loads ONNX models trained from the technical-feature pipeline and executes weighted-voting trades inside MT5.

- `MacroNewsOnnxTrader.mq5`
  Loads ONNX models trained from the macro-news pipeline and trades only when relevant macroeconomic events are released.

- `Assets/`
  Runtime bundle for the EAs, including ONNX model files and configuration CSV files.

## Main Trading Modes

### 1. Technical ensemble execution

`EnsembleOnnxTrader.mq5` is used for the technical-feature workflow. It supports the five model outputs used in the study:

- Logistic Regression
- SVM
- Random Forest
- XGBoost
- Ensemble

The EA reads exported feature configuration files and ONNX models from `Assets/`, reconstructs the feature vector inside MT5, and generates trading decisions from the selected model or the weighted ensemble.

### 2. Macro-news-driven execution

`MacroNewsOnnxTrader.mq5` is used for the macroeconomic-news workflow. It reads:

- `macro_config_<pair>.csv`
- `macro_runtime_events_<pair>.csv`
- `macro_<model>_<pair>.onnx`

The EA replays event-aligned features generated during training and executes trades when a relevant scheduled macroeconomic event is encountered during backtesting.

## Supported Currency Pairs

The bundled assets currently cover these pairs:

- `EURUSD`
- `GBPUSD`
- `AUDUSD`
- `EURJPY`
- `USDJPY`

## Notes

- Source files are included in this repository.
- Compiled MT5 binaries such as `.ex5` are intentionally excluded.
- Large raw datasets and legacy local folders are also excluded from GitHub, but they remain available in the local workspace.
- The active runtime bundle is `Assets/`. The legacy folder `EnsembleOnnxTrader_Assets/` is kept locally only and is not part of the published snapshot.
