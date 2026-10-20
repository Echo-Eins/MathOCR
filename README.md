# MathOCR — изображение формулы → LaTeX

Нейросетевая модель оптического распознавания печатных математических
формул: CNN + Transformer encoder-decoder на PyTorch (42,9 M параметров),
обученная детерминированно на зафиксированной версии данных
(im2latex-100k + arXiv-срез OleehyO/latex-formulas, ~205 тыс. пар) и
проверенная на внешнем источнике (UniMER-Test).

**Детерминированность**: одно и то же изображение всегда даёт одну и ту же
формулу (канонический путь инференса), а один и тот же config + seed —
побитово одни и те же веса, в том числе после прерывания и `--resume`.
Подробности, обоснования и измерения — [`docs/DESIGN.md`](docs/DESIGN.md).

## Быстрый старт

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e . pytest
export PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8

# 1. Данные: скачивание закреплённых ревизий, нормализация, разбиения (~10 мин, 4 CPU)
.venv/bin/python -m mathocr.data.build --config configs/base.yaml
#    -> data/processed/{train,val,test_*,ood_*}.parquet, manifest.json, vocab.json

# 2. Один seed целиком: обучение -> метрики -> CDM
scripts/run_seed.sh 1
#    повторный запуск продолжает прерванное обучение с границы эпохи

# 3. Среднее ± std по seed'ам
.venv/bin/python -m mathocr.aggregate runs/base_s1 runs/base_s2 runs/base_s3 --out results/base

# Распознать картинки
.venv/bin/python -m mathocr.predict --checkpoint runs/base_s1/checkpoints/best.pt formula.png
```

Нужны: Python 3.12, PyTorch ≥ 2.6 (CUDA), Node.js (нормализатор LaTeX при
сборке данных), docker (только для CDM: `docker build -f docker/cdm.Dockerfile -t mathocr-cdm docker/`).

## Структура

| Путь | Что |
|---|---|
| `configs/base.yaml` | единственный источник параметров: данные, предобработка, модель, обучение |
| `src/mathocr/data/sources.py` | ревизии HF и sha256 исходных файлов |
| `src/mathocr/data/build.py` | разбиения, дедупликация, фильтры, манифест |
| `src/mathocr/normalize.py` | приведение сырого LaTeX к конвенции im2markup |
| `src/mathocr/image.py` | каноническая предобработка изображения |
| `src/mathocr/model.py` | модель, жадное декодирование с KV-кэшем |
| `src/mathocr/train.py` | обучение одного seed'а |
| `src/mathocr/evaluate.py`, `cdm.py`, `aggregate.py` | EM / Edit distance / BLEU, CDM, сводка по seed'ам |
| `src/mathocr/predict.py` | CLI распознавания |
| `third_party/latex_normalizer` | нормализатор im2markup/KaTeX из UniMERNet (лицензии внутри) |
| `tests/` | тесты, включая побитовую воспроизводимость обучения |

## Состояние работы

Что сделано, результаты всех прогонов и проб, как развернуть и с чего
продолжать — [`docs/HANDOFF.md`](docs/HANDOFF.md); отложенные идеи —
[`tasks/todo.md`](tasks/todo.md).

## Результаты

Заполняются по мере прогонов (`results/`).
