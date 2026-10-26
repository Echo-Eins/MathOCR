# Чтение страниц: датасеты, бенчмарки, синтетика (обзор 2026-10-27)

Метки: [V] проверено на первоисточнике, [S] только вторичный источник, [U] не проверено. Обзор — веб-поиск
(агент), без загрузок; лицензии и объёмы перед использованием сверять заново.

## Главное
- Открытых данных «страница + полный текст» в масштабе миллионов — много, но метки **машинные**: текстовый слой PDF,
  OCR-движок (AWS Textract / Tesseract) или крупная модель-учитель (GPT-4.1, Qwen3-VL-235B). **Ручных** полных
  страниц мало: печать EN < ~10 тыс.; рукопись EN ~35 тыс. отобранных + краудсорсинг Library of Congress «By the
  People» ~1 млн страниц (текст CC0, без рамок строк).
- Точнее всего — метки из исходника (рендер LaTeX / HTML / шрифтами) → синтетика с точными метками обязательна.
- Целевой бенчмарк печати: **olmOCR-Bench** (EN, юнит-тесты, ODC-BY). Планка в классе ≤ 300M: **Falcon OCR 1.5
  (TII, 270M, Apache-2.0) — 82,4** (7 категорий без колонтитулов); под 1B — LightOnOCR-3-0.8B 85,5 [S]; лидеры ≈ 87–88.
  Только текстом весь бенчмарк не выиграть (математика — 2 из 8 категорий, таблицы — 1) — сначала по категориям.
- Рукопись: в olmOCR-Bench её нет → IAM paragraph (VAN 4,45 % CER), GNHK, Bentham, SMHD; страничного рейтинга для
  малых моделей нет — заявка «первые на всех четырёх» нова.

## Обучающие источники (печать)
| источник | что | метки | лицензия | заметки |
|---|---|---|---|---|
| [pixparse/pdfa-eng-wds](https://huggingface.co/datasets/pixparse/pdfa-eng-wds) [V] | 2,16 M PDF ≈ 18 M страниц, рендер свой (300 dpi) | текстовый слой PDF, слова/строки с рамками | Common Crawl / Digital Corpora terms | 1,57 TB; моджибейк, текст-картинки теряются |
| [lightonai/LightOnOCR-mix-0126](https://huggingface.co/datasets/lightonai/LightOnOCR-mix-0126) [V] | 16,43 M страниц PDFA (только метки, картинки — из pdfa-eng-wds по `key` / `page_idx`) | Qwen3-VL-235B → Markdown + LaTeX + HTML-таблицы | other (PDFA terms) | крупнейший открытый источник хороших целей страница → Markdown |
| [allenai/olmOCR-mix-1025](https://huggingface.co/datasets/allenai/olmOCR-mix-1025) [V] | 270 тыс. страниц PDF (документы 233 тыс., книги 17 тыс., **LoC 10 тыс., NARA 10 тыс. рукописей**) | GPT-4.1; LoC/NARA — ручные транскрипты, почищенные ChatGPT | ODC-BY | 77 GB, PDF внутри |
| [allenai/olmOCR-synthmix-1025](https://huggingface.co/datasets/allenai/olmOCR-synthmix-1025) [V] | 2,2 тыс. страниц из HTML + 30 тыс. юнит-тестов | точные | ODC-BY | шаблон для RL с наградой-тестом |
| [pixparse/idl-wds](https://huggingface.co/datasets/pixparse/idl-wds) [V] | **сканы**, 19,2 M страниц | AWS Textract | other | 2,6 TB; старые факсы, формы |
| [Infinity-Doc-400K](https://huggingface.co/datasets/infly/Infinity-Doc-400K) [V] | 325 тыс. реальных сканов + 62 тыс. синтетики | псевдометки нескольких моделей / точные | **CC-BY-NC-SA, gated** | качество меток реальных — «довольно низкое» |
| [HuggingFaceFW/finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) [V] | только текст + URL PDF (EN 207 M док., **RU 16 M**) | docling / rolmOCR | ODC-By | PDF качать самим |
| [DocLayNet v1.2](https://huggingface.co/datasets/ds4sd/DocLayNet-v1.2) [V], [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m) [V] | 81 тыс. / 575 тыс. страниц | ручная разметка вёрстки + текст PDF / все слова страницы | CDLA-Permissive | PubTables-1M — недооценённый источник полного текста страниц |
| [DocGenome](https://huggingface.co/datasets/U4R/DocGenome) [V] | 6,8 M страниц arXiv | LaTeX каждого блока + порядок | CC-BY-4.0 (+ лицензии статей) | страничный LaTeX собирается |
| [SynthDoG-en](https://huggingface.co/datasets/naver-clova-ix/synthdog-en) [V] + генератор (MIT, репозиторий Donut) | синтетика | точные | MIT | генератор легко переносится на RU |

Обучающих данных лидеры (PaddleOCR-VL, MinerU2.5, dots.ocr, DeepSeek-OCR, GOT-OCR2, GLM-OCR, Nanonets, HunyuanOCR) не
публикуют — только веса.

## Рукопись (первоклассная цель)
| набор | страниц / авторов | стиль | метки | лицензия | роль |
|---|---|---|---|---|---|
| [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) [V] | 1 539 форм, 657 авторов | аккуратная печатно-полукурсивная | строки + слова XML | **NC** | тест: paragraph test 336 стр. |
| [CVL](https://zenodo.org/records/1492267) [V] | ~1 604 стр., 310 авторов, 7 текстов | смешанная | слова | CC BY-NC | только тест разнообразия авторов |
| [Bentham](https://zenodo.org/record/44519) + [HTRtS 2015](https://zenodo.org/record/248733) [V] | 433 стр. | трудная курсива XVIII–XIX вв., вычёркивания | PAGE XML | **CC BY 4.0** | обучение + тест |
| [Revolutionary City](https://zenodo.org/records/15776323) [V] | **3 316 стр.**, 96 тыс. строк, много рук, 1758–1805 | историческая курсива | ALTO | **CC BY 4.0** | лучший открытый исторический EN |
| [JCRS](https://zenodo.org/records/4243023) [V] | 2 660 стр., 1900–1950 | медицинская | строки | CC BY 4.0 | обучение |
| [GNHK](https://huggingface.co/datasets/aryaman27/goodnotes-handwritten-dataset) [V/S] | 687 **фото** конспектов (515 / 172) | неряшливая, смешанная, с математикой | слова + `line_idx` | **CC BY 4.0** | обучение + ключевой тест |
| [IIIT-HW-English-Word](https://cvit.iiit.ac.in/usodi/bgroehtr.php) [V] | **20 800 стр. A4, 1 215 авторов** (фото / сканер) | современная | слова JSON | **не указана** — спросить разрешение | крупнейший современный набор (страницы короткие) |
| SMHD (RMIT 2023) [V] | > 500 студентов | очень неряшливая | текст страницы | CC BY-NC | тест неряшливой рукописи |
| [school_notebooks_EN](https://huggingface.co/datasets/ai-forever/school_notebooks_EN) [V] | 90 фото 3000×4000 | тетради | полигоны + текст | MIT | тест |
| **[LoC By the People](https://github.com/libraryOfCongress/btp-data)** [V] | ~725 тыс. стр. (2023), > 1 M (2025) [S]; тысячи авторов, письма и дневники XVIII–XX вв. | рукопись + машинопись | **CC0** текст, переносы строк сохранены, `[?]`, вычёркивания | права на картинки — по коллекциям; loc.gov режет быстрые загрузки | единственный крупный источник реальной рукописи |

Синтез рукописи: [Emuru](https://huggingface.co/blowing-up-groundhogs/emuru) (CVPR 2025, строки, стиль по одному
образцу, **MIT, обучен только на рендерах шрифтов**; не гарантирует текст — фильтровать распознавателем);
DiffBrush, DiffusionPen, One-DM, VATr, Paragraph-LDM — на IAM → только для исследований. Шрифты: **Google Fonts,
360 рукописных семейств (OFL / Apache)**, 1001fonts — 6 872 бесплатных для коммерции (проверять). Деградации:
[Augraphy](https://github.com/sparkfish/augraphy) (MIT). Перенос: одна синтетика проигрывает реальным (VATr: IAM CER 27
против 5), синтетика + дообучение на 20 % реальных — 73,8 → 6,5 CER; DAN: страничная синтетика + учебный план от одной
строки к целой странице.

## Фото документов
Реальных фото с текстом мало: [Brno Mobile OCR](https://pero.fit.vutbr.cz/brno_mobile_ocr_dataset) (19,7 тыс. фото
2,1 тыс. страниц, 23 телефона; лицензия противоречива), [SmartDoc 2015](https://zenodo.org/record/2572929) (50
документов). Наборы выпрямления (DocUNet, UVDoc, Doc3D, …) — без ручного текста. **Фото разворотов книг / тетрадей с
транскриптом не найдено** — делать самим: печатать отрисованные страницы, фотографировать, переносить метки
([DocAligner](https://arxiv.org/abs/2306.05749)).

## Бенчмарки
| бенчмарк | что | метрика | лицензия | планка малых |
|---|---|---|---|---|
| **[olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench)** | 1 403 EN PDF, 7 010 тестов (arXiv-математика, таблицы, старые сканы, колонтитулы, многоколоночный, мелкий длинный текст) | юнит-тесты, макросреднее | ODC-BY | Falcon OCR 1.5 (270M) 82,4; LightOnOCR-3-0.8B 85,5 |
| [OmniDocBench v1.6](https://github.com/opendatalab/OmniDocBench) | 1 651 стр. EN + ZH, 10 типов (в т. ч. рукописные заметки) | 1 − edit, TEDS, CDM | **только исследования** | Dolphin-1.5 (0,3B) 86,52; лидеры ~96–97; насыщен, ~12 % блоков с ошибками ([PureDocBench](https://arxiv.org/abs/2605.07492)) |
| [Real5-OmniDocBench](https://huggingface.co/datasets/JiangWenxiang/Real5-OmniDocBench) | те же страницы × 5 условий съёмки | как OmniDocBench | от исследовательского | лучший < 1B ≈ 61,8 — **низкая планка** |
| [Fox](https://github.com/ucaslcl/Fox) | 112 EN плотных стр. | edit, F1 | CC BY-NC | GOT 0,035 — проверка на здравость |

## Предлагаемая раскладка 200 тыс. обучающих пар (из обзора, к обсуждению)
| источник | страниц |
|---|---|
| печать born-digital (LightOn / PDFA) | 55 тыс. |
| olmOCR-mix (документы, книги) | 30 тыс. |
| сканы (IDL / Infinity) | 20 тыс. |
| arXiv / математика (точные) | 15 тыс. |
| синтетика печати + развороты | 15 тыс. |
| реальная рукопись (LoC 30 тыс., olmOCR LoC/NARA 15 тыс., отобранные ~8 тыс., IIIT-HW при разрешении) | 45–55 тыс. |
| синтетическая рукопись | 20–25 тыс. |
+ ~15 % в «фото»-версии (печать и съёмка + аугментация).

**100 тыс. на проверку**: ручного «золота» EN всего ~10 тыс. → остальное — отложенное «серебро» / синтетика,
разбиение по документу, автору, кампании LoC (без утечек); золотое ядро для главных чисел: IAM paragraph test, GNHK test,
Bentham test, SMHD, Washington, срез Revolutionary City, Real5, английская часть OmniDocBench. Дедупликация обучающих PDF
против olmOCR-Bench и OmniDocBench (URL, хэш); на Real5 / DocPTBench не учиться (они из OmniDocBench).

## Лицензии
Некоммерческие: IAM, CVL, Washington, SMHD, CHURRO, OmniDocBench и Real5, Infinity-Doc-400K, HKR, генераторы на IAM.
Неясные: IIIT-HW, B-MOD, PDFA / IDL, ReadingBank. Чистые: LoC By the People (текст), Bentham, Revolutionary City, GNHK,
olmOCR-mix, PubTables-1M, DocLayNet, Emuru, Google Fonts.

## Русский (на потом)
[school_notebooks_RU](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) (MIT, фото, многие — развороты),
[HWR200](https://huggingface.co/datasets/AntiplagiatCompany/HWR200) (200 авторов, 30 тыс. изображений, Apache-2.0),
[Digital Peter](https://huggingface.co/datasets/ai-forever/Peter), HKR (NC-ND); FinePDFs `rus_Cyrl`; тест
[ru-ocr-benchmark-hard](https://huggingface.co/datasets/metrika-labs/ru-ocr-benchmark-hard) (155 стр.).
