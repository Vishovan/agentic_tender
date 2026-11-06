# Ручная проверка отложенной выборки (`held_out_draft.json`)

Мы вручную вычитали и закрыли отложенную выборку в [held_out_draft.json](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json): проверили все 15 пар «требование × артикул» (`G01`–`G15`) и список требований первой позиции из прозового техзадания закупки `02` (`E01`). Каждую запись сверяли не по пересказу пайплайна, а напрямую по первичным файлам датасета — таблицам Word и Excel, XML-извещению ЕИС, PDF техзадания и карточкам четырёх помп.

Ниже собрано, зачем эта выборка была нужна, какие числа получились у базового конвейера и какие нетривиальные случаи вскрылись при проверке глазами.

---

## 1. Зачем делали эту разметку и что в неё вошло

До этого шага мы могли сравнивать версии пайплайна только друг с другом (так закрылись гипотезы `H1`, `H3`, `H6`, `H9` и `H10` в [experiments.py](file:///Users/user/Developer/agentic_tender/src/tenderai/experiments.py)). Но три оставшиеся гипотезы без человеческого эталона проверить нельзя:
- **`H2` (эмбеддинги против поиска по словам)** упирается в вопрос: *ту ли вообще строчку в карточке SKU нашёл поиск?*
- **`H7` и `H8` (кодовый чекер против вердикта LLM и каскада)** требуют независимого ответа `match / mismatch / unknown`, чтобы посчитать честный `macro-F1` и точность класса `match`.

Поэтому в [held_out_draft.json](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json) собраны две части:
1. **15 пар `G01`–`G15`** из четырёх разных закупок (`01`, `02`, `04`, `07`) и трёх видов источников: DOCX-таблицы ([03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx)), XLSX-таблицы ([04_Описание объекта закупки.xlsx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/04_0128200000125006876/04_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%D0%B0%20%D0%B7%D0%B0%D0%BA%D1%83%D0%BF%D0%BA%D0%B8.xlsx)), XML-извещения ЕИС ([04_Извещение.xml](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/07_0372100048826000117/04_%D0%98%D0%B7%D0%B2%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D0%B5.xml)) и PDF с прозой ([05_Техническое_задание_16568494-1.pdf](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/02_0373200087824002359/05_%D0%A2%D0%B5%D1%85%D0%BD%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D0%B7%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5_16568494-1.pdf)).
2. **Блок `E01`** — полный список технических требований к Позиции № 1 на 5-й странице прозового PDF закупки `02`, извлечённый через `gpt-5.6-luna`.

---

## 2. Главный сюрприз в цифрах: почему общая гистограмма статусов обманчива

Если посмотреть только на общее распределение статусов по 15 парам, кажется, что базовый лексический пайплайн отработал без единой помарки:

- **Ответы базового пайплайна (`pipeline_status`)**: `7 match`, `4 mismatch`, `4 unknown`.
- **Ручной эталон (`gold_status`)**: те же самые `7 match`, `4 mismatch`, `4 unknown`.

Но внутри этих одинаковых сумм скрылись сразу две встречные ошибки и пять промахов выравнивания:

1. **Две ошибки статуса взаимно скомпенсировались**:
   - в **`G11`** пайплайн поставил **ложный `match`** там, где на самом деле `unknown`;
   - в **`G12`** пайплайн поставил **`unknown`** там, где на самом деле чистый `match`.
2. **Лексический поиск попал в нужную характеристику только в 10 парах из 15 (`66.7%`)**:
   - в пяти случаях (`G08`, `G11`, `G12`, `G13`, `G14`) функция [`align`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L58-L92) либо вообще ничего не нашла, либо притянула чужую строку из карточки товара.
   - в трёх из этих пяти случаев (`G08`, `G13`, `G14`) итоговый статус случайно совпал с эталоном — дважды выручили защитные проверки в [`check_constraints`](file:///Users/user/Developer/agentic_tender/src/tenderai/checker.py#L11-L18), а один раз (`G14`) совпал сам знак ответа, но цитата со стороны товара оказалась слишком общей и не доказывала все детали требования.

Отсюда получаются честные стартовые метрики нашего baseline на отложенной выборке (точка отсчёта для `H2`, `H7` и `H8`):

| Метрика из [TASK.md](file:///Users/user/Developer/agentic_tender/TASK.md#L43-L50) | Значение baseline | Пояснение |
|---|---|---|
| Точность выравнивания характеристик (`gold_alignment_ok`) | **10 / 15 (`0.667`)** | Промахи на `G08`, `G11`, `G12`, `G13`, `G14` |
| Совпадение статусов (accuracy) | **13 / 15 (`0.867`)** | Перепутаны `G11` (ложный `match`) и `G12` (пропущенный `match`) |
| `macro-F1` по трём классам | **`0.869`** | `F1(match) = 0.857`, `F1(mismatch) = 1.000`, `F1(unknown) = 0.750` |
| Precision класса `match` | **6 / 7 (`0.857`)** | 1 ложный `match` на `G11` |
| Доказательность решений `match` | **5 / 7 (`0.714`)** | Кроме ложного `G11`, в `G14` верный `match` опёрся на неполную цитату SKU |
| Извлечение требований Luna на `E01` | **12 / 12 (`1.000`)** | Пропущенных и лишних требований на стр. 5 нет |

---

## 3. Подробный разбор интересных случаев

### Случай 1 (`G11`). Тихий ложный `match`: номинальный объём против максимального
- **Карточка выборки**: [`G11` в held_out_draft.json](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L219-L237).
- **Откуда требование**: закупка `01`, файл [03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx), таблица 1, строка 5 (`table-1!D5`, Позиция № 1; в JSON — [requirements.json, строки 56–81](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/requirements.json#L56-L81)):
  > *«Номинальный объем заполнения: ≥ 300 и ≤ 399»* (мл).
- **Что реально в карточке `CT0050L`**: в [CT0050L.xlsx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/SKU/CT0050L.xlsx) и [skus.json](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json) про объём записаны четыре разные строки:
  1. `Характеристики!A5` ([skus.json, строки 3–28](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L3-L28)): `Номинальный объём резервуара = 250 мл`.
  2. `Характеристики!A6` ([skus.json, строки 30–55](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L30-L55)): `Допустимый объём заполнения = Не указано`.
  3. `Характеристики!A17` ([skus.json, строки 327–352](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L327-L352)): `Максимальный объём заполнения = 300 мл`.
  4. `Характеристики!A18` ([skus.json, строки 354–379](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L354-L379)): `Объём заполнения в таблице производителя = 240 мл`.
  В самой цитате французского каталога (`2 jours | 240 ml | 250 ml | 300 ml | 5 | CT0050L`) прямо видно, что это помпа на **250 мл** (штатное заполнение на двое суток — 240 мл), а **300 мл** — это верхний предел, выше которого резервуар нельзя переливать.
- **Почему ошибся пайплайн**: правило `fill_volume` в [`canon_key`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L16) ищет только корни `("объем", "заполнен")` и не отличает *номинальный* объём от *максимального*. Строку `A6` («Допустимый объём заполнения») фильтр [`_comparable`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L40-L42) пропустил, потому что там стоит `Не указано`, и взял следующую подходящую по ключу строку — `Максимальный объём заполнения = 300`. Число `300` попало на границу требуемого интервала `[300, 399]`, и чекер выдал ложный `match`.
- **Решение в эталоне**: `gold_alignment_ok = false`, `gold_status = "unknown"` (номинальный объём заполнения в карточке отдельно не установлен, а подменять его предельным переливом нельзя).

---

### Случай 2 (`G12`). Потерянный `match`: разные слова об одном и том же свойстве
- **Карточка выборки**: [`G12` в held_out_draft.json](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L239-L257).
- **Откуда требование**: закупка `01`, [03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx), строка 6 (`table-1!D6`, Позиция № 1; [requirements.json, строки 83–108](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/requirements.json#L83-L108)):
  > *«Регулирование скорости инфузии: нет»*.
- **Что в карточке `CT0050L`**: строка `Характеристики!A7` ([skus.json, строки 57–82](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L57-L82)):
  > *«Тип скорости: Постоянная»*.
- **Что здесь интересно**: по смыслу «постоянная скорость» и «регулирования скорости нет» — это одно и то же. Более того, автор кодового чекера заранее предусмотрел эту пару в [`_FALSE_TEXT`](file:///Users/user/Developer/agentic_tender/src/tenderai/checker.py#L7): слово `"постоянная"` там лежит рядом со словом `"нет"`. Если бы эти две строки встретились в [`check_constraints`](file:///Users/user/Developer/agentic_tender/src/tenderai/checker.py#L24-L25), чекер сразу вернул бы `match`.
- **Почему пара потерялась**: лексический поиск в [`align`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L13-L14) повесил на требование ярлык `speed_control` (корни `"регулир"` + `"скорост"`), а на строку SKU `"Тип скорости"` — другой ярлык `infusion_speed` (только `"скорост"`). Запасной поиск по пересечению слов ([`_tokens`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L54-L55)) насчитал всего одно общее слово из трёх (`скорости` из `{регулирование, скорости, инфузии}`, коэффициент Жаккара `0.33` при пороге `0.5`) и отбросил пару в `not_aligned`. Это самый наглядный кандидат на починку через плотные эмбеддинги в гипотезе `H2`.
- **Решение в эталоне**: `gold_alignment_ok = false`, `gold_status = "match"`.

---

### Случай 3 (`G05` в паре с `G04`). Близко по смыслу, но несовместимо по числу
- **Карточки выборки**: [`G04`](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L79-L97) и [`G05`](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L99-L117).
- **Откуда требование**: закупка `01`, [03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx), строка 9 (`table-1!D9`; [requirements.json, строки 164–189](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/requirements.json#L164-L189)):
  > *«Скорость инфузии, мл/час: 5»*.
- **Сравнение двух мультискоростных помп**:
  - У **`M275-07`** (`G04`, строка `Характеристики!A8`, [skus.json, строки 1056–1090](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L1056-L1090)) шкала регулятора идёт с шагом 1 мл/ч: `0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 мл/ч`. Пятёрка в наборе есть → `match` (`value_in_sku_set`).
  - У **`MS5248M`** (`G05`, строка `Характеристики!A8`, [skus.json, строки 2645–2678](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L2645-L2678)) шкала идёт с шагом 2 мл/ч, только по чётным числам: `2 / 4 / 6 / 8 / 10 / 12 / 14 мл/ч`. Есть соседние скорости `4` и `6`, но режима `5 мл/ч` на переключателе нет → `mismatch` (`value_not_in_sku_set`).
- **Зачем этот случай нужен на защите**: задание ([TASK.md, строка 41](file:///Users/user/Developer/agentic_tender/TASK.md#L41)) прямо просит показать пару, которая очень близка в векторном пространстве, но технически не подходит. Любой текстовый энкодер положит «Скорость инфузии 5 мл/ч» и «Настройки скорости 2/4/6/8/10/12/14 мл/ч» почти в одну точку, а кодовая проверка вхождения числа во множество ([`_contains`](file:///Users/user/Developer/agentic_tender/src/tenderai/checker.py#L36-L39)) без колебаний разводит `M275-07` и `MS5248M` по разным вердиктам.
- **Решение в эталоне (`G05`)**: `gold_alignment_ok = true`, `gold_status = "mismatch"`.

---

### Случай 4 (`G14`). Семь скоростей, режим «0» и почему черновик зря сомневался в восьмой ступени
- **Карточка выборки**: [`G14` в held_out_draft.json](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L279-L297).
- **Откуда требование**: закупка `01`, [03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx), таблица 1, строка 36 (`table-1!D36`, Позиция № 2 — второй насос в том же документе; [requirements.json, строки 893–918](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/requirements.json#L893-L918)):
  > *«7-мь вариантов выбора скорости с возможностью немедленной остановки инфузии (режим "0"): Наличие»*.
- **Как мы распутали этот случай при проверке**:
  1. В предварительной заметке `note` черновика было высказано сомнение: у `M275-07` набор `0 / 1 / 2 / 3 / 4 / 5 / 6 / 7` содержит 8 чисел, а не 7. Но если посмотреть на две строки выше в самом ТЗ ([03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx), строка 34, `Скорость инфузии: 2:4:6:8:10:12:14`), там перечислено ровно **7 рабочих скоростей**, а положение `"0"` в медицинской формулировке идёт отдельно как режим немедленной остановки.
  2. Открываем карточку [M275-07.xlsx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/SKU/M275-07.xlsx) и видим прямое подтверждение обеих частей: в строке `Характеристики!A16` ([skus.json, строки 1281–1315](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L1281-L1315)) записано `Рабочие скорости без режима «0» = 1 / 2 / 3 / 4 / 5 / 6 / 7` (ровно **7 рабочих скоростей**), а в строке `Характеристики!A9` ([skus.json, строки 1092–1117](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L1092-L1117)) — `Режим остановки «0» = Да`.
  3. Значит, само это требование (конструкцию переключателя на 7 рабочих ступеней плюс стоп-режим `"0"`) помпа `M275-07` выполняет честно. А то, что сами числа скоростей в Позиции № 2 нужны другие (`2..14`, а не `1..7`), проверяется соседним атомарным требованием из строки 34 ([checks.json, `85f942d5ea61`](file:///Users/user/Developer/agentic_tender/runs/h4/comparisons/checks.json)), которое и переводит всю Позицию № 2 в `mismatch`.
- **Почему `gold_alignment_ok` всё равно `false`**: пайплайн поставил `match`, но привязался не к строкам `A8/A9/A16`, а к общей строке `Характеристики!A7` (`Тип скорости = Регулируемая`, [skus.json, строки 1029–1054](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L1029-L1054)), потому что фильтр [`_comparable`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L47-L48) не разрешает сравнивать булево слово `"Наличие"` со списком чисел `kind = "set"`. Одно слово `"Регулируемая"` не доказывает ни число ступеней, ни наличие нуля — будь у помпы три скорости без режима `"0"`, пайплайн так же поставил бы `match`.
- **Решение в эталоне**: `gold_status = "match"`, но `gold_alignment_ok = false`.

---

### Случай 5 (`G08` и `G13`). Когда поиск по подстрокам цепляется за случайные слова, а чекер страхует от беды

Оба примера показывают слабость жёстких правил в [`canon_key`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L11-L34) на длинных формулировках и одновременно объясняют, зачем в чекере стоят защитные проверки перед сравнением значений:

- **`G08` — индивидуальная упаковка против газовой стерилизации** ([held_out_draft.json, строки 159–177](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L159-L177)):
  - В закупке `04` ([04_Описание объекта закупки.xlsx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/04_0128200000125006876/04_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%D0%B0%20%D0%B7%D0%B0%D0%BA%D1%83%D0%BF%D0%BA%D0%B8.xlsx), ячейка `Лист14!E17`; [requirements.json, строки 1623–1648](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/requirements.json#L1623-L1648)) написано: *«Упаковка индивидуальная стерильная: соответствие»*.
  - В карточке [CT0050L.xlsx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/SKU/CT0050L.xlsx) есть количество штук в коробке (`12`), факт стерильности изделия (`Характеристики!A28`) и способ стерилизации (`Mode de stérilisation : oxyde d’éthylène`, [skus.json, строки 624–649](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L624-L649) и [759–784](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L759-L784)), но нет ни слова про индивидуальный пакет на каждую помпу.
  - Правило `("sterile", ("стериль",))` в [`align.py`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L23) увидело корень `стериль` и привязало упаковку к стерилизации оксидом этилена (`gold_alignment_ok = false`). От ложного срабатывания спасло то, что слово `"соответствие"` разобралось как `kind = "reference"`, и [`check_constraints`](file:///Users/user/Developer/agentic_tender/src/tenderai/checker.py#L12-L13) честно остановился со статусом `unknown` (`requirement_unparsed`).

- **`G13` — хвост длинной фразы утаскивает поиск в объём резервуара** ([held_out_draft.json, строки 259–277](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L259-L277)):
  - В закупке `01` ([03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx), строка 14, `table-1!D14`; [requirements.json, строки 300–325](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/requirements.json#L300-L325)) требуют: *«Механизм, контролирующий точность подачи раствора, и обеспечивающий гарантию точности инфузии при заполнении резервуара от 50% от объема резервуара: Наличие»*.
  - В [`canon_key`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py#L15) сработало правило `("reservoir_volume", ("объем", "резервуар"))` — просто потому, что оба слова встретились в придаточном обороте на конце предложения. В результате требование про механизм точности соединилось с ячейкой `Номинальный объём резервуара = 250 мл` (`gold_alignment_ok = false`).
  - Здесь сработала уже защита единиц измерения из гипотезы `H10` ([`check_constraints`, строки 16–17](file:///Users/user/Developer/agentic_tender/src/tenderai/checker.py#L16-L17)): парсер вытащил из `50%` единицу `pct`, увидел конфликт с `ml` у `250 мл` и вернул `unknown` (`unit_incompatible`). Поскольку у `CT0050L` данных о таком механизме действительно нет (эта строка есть только у `MS5248M` и `MC1248L`), эталонный статус тоже `unknown`.

---

### Случай 6 (`G07`, `G15` и блок `E01`). Пустые ячейки SKU и сквозная проверка прозы из GPT-Luna

- **`G07` — отсутствие данных не равно отказу** ([held_out_draft.json, строки 139–157](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L139-L157)):
  - Закупка `01` требует `Защита от УФ-лучей: Да` ([03_Описание.docx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/01_0321200014126000201/03_%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5.docx), строка 4). В карточке [M275-07.xlsx](file:///Users/user/Developer/agentic_tender/dataset_zakupki/SKU/M275-07.xlsx) на строке `Характеристики!A13` ([skus.json, строки 1200–1225](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L1200-L1225)) стоит `Не указано` (`knowledge = "not_established"`, цитата `—`). И поиск, и чекер отработали строго по правилам задания: характеристику нашли (`gold_alignment_ok = true`), но вместо `mismatch` вернули `unknown` (`missing_sku_value`).

- **`G15` и `E01` — проверка того, что достала `gpt-5.6-luna` из PDF закупки `02`** ([held_out_draft.json, строки 299–342](file:///Users/user/Developer/agentic_tender/eval/held_out_draft.json#L299-L342)):
  - В [05_Техническое_задание_16568494-1.pdf](file:///Users/user/Developer/agentic_tender/dataset_zakupki/tenders/02_0373200087824002359/05_%D0%A2%D0%B5%D1%85%D0%BD%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5_%D0%B7%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5_16568494-1.pdf) (текстовый дамп — [02_tz.txt, строки 245–272](file:///Users/user/Developer/agentic_tender/runs/prose/02_tz.txt#L245-L272)) всего описаны **6 позиций помп** на страницах 5–9, и в основном прогоне модель извлекла все 6 позиций целиком — 65 требований в [0373200087824002359_sdk_requirements.json](file:///Users/user/Developer/agentic_tender/runs/luna/0373200087824002359_sdk_requirements.json).
  - Для ручной контрольной выборки в `E01` взята Позиция № 1 со страницы 5. Сверка по тексту [02_tz.txt (строки 246–266)](file:///Users/user/Developer/agentic_tender/runs/prose/02_tz.txt#L246-L266) показала стопроцентное попадание: все 12 технических условий выписаны дословно, ни одно не потеряно, а адрес доставки (*п. Коммунарка, ул. Сосенский Стан*) и сроки поставки (*по 19.12.2025*) модель сама отфильтровала и в технические требования не потащила.
  - В карточке `G15` одно из этих извлечённых требований (`Болюс: Нет.` со стр. 5) проходит дальше в кодовый чекер против `MS5248M` (`Болюс = Да`, [skus.json, строки 2707–2732](file:///Users/user/Developer/agentic_tender/runs/h4/prepared/skus.json#L2707-L2732)) и штатно получает `mismatch`.

---

## 4. Итоговая сводка по всем 15 парам (Лексический baseline vs `FRIDA + RRF`)

| ID | Закупка (источник) | SKU | Требование | Лексический baseline (`align_ok`) | Итог `FRIDA + RRF` (`align_ok`) | Эталон (`gold`) |
|---|---|---|---|---|---|---|
| `G01` | `01` (DOCX, поз. 1) | `MS5248M` | `Болюс: Нет` | `mismatch` (`true`) | `mismatch`, `cos=0.816` (`true`) | `mismatch` |
| `G02` | `01` (DOCX, поз. 1) | `CT0050L` | `Болюс: Нет` | `match` (`true`) | `match`, `cos=0.816` (`true`) | `match` |
| `G03` | `01` (DOCX, поз. 1) | `CT0050L` | `Скорость, мл/час: 5` | `match` (`true`) | `match`, `cos=0.768` (`true`) | `match` |
| `G04` | `01` (DOCX, поз. 1) | `M275-07` | `Скорость, мл/час: 5` | `match` (`true`) | `match`, `cos=0.768` (`true`) | `match` |
| `G05` | `01` (DOCX, поз. 1) | `MS5248M` | `Скорость, мл/час: 5` | `mismatch` (`true`) | `mismatch`, `cos=0.768` (`true`) | `mismatch` |
| `G06` | `01` (DOCX, поз. 2) | `CT0050L` | `Ном. объем резервуара: 100..124` | `mismatch` (`true`) | `mismatch`, `cos=0.836` (`true`) | `mismatch` |
| `G07` | `01` (DOCX, поз. 1) | `M275-07` | `Защита от УФ-лучей: Да` | `unknown` (`true`) | `unknown`, `cos=0.816` (`true`) | `unknown` |
| `G08` | `04` (XLSX, поз. 1) | `CT0050L` | `Упаковка индив. стерильная` | `unknown` (**`false`**, чужой слот) | `unknown` (**`true`**, `not_aligned`) | `unknown` |
| `G09` | `04` (XLSX, поз. 1) | `CT0050L` | `Система блокировки болюса: Нет` | `match` (`true`) | `match`, `cos=0.739` (`true`) | `match` |
| `G10` | `07` (XML ЕИС) | `CT0050L` | `Болюс: Нет` | `match` (`true`) | `match`, `cos=0.816` (`true`) | `match` |
| `G11` | `01` (DOCX, поз. 1) | `CT0050L` | `Ном. объем заполнения: 300..399` | **`match` (`false`, Макс.=300)** | **`unknown` (`true`, фильтр квал.)** | **`unknown`** |
| `G12` | `01` (DOCX, поз. 1) | `CT0050L` | `Регулирование скорости: нет` | **`unknown` (`false`, `null`)** | **`match`, `cos=0.636` (`true`)** | **`match`** |
| `G13` | `01` (DOCX, поз. 1) | `CT0050L` | `Механизм точности от 50% объёма` | `unknown` (**`false`**, 250 мл) | `unknown` (**`true`**, `not_aligned`) | `unknown` |
| `G14` | `01` (DOCX, поз. 2) | `M275-07` | `7 вариантов скорости + режим "0"` | `match` (**`false`**, Регулируемая) | `unknown` (`false`, обе строки в Top-3) | `match` |
| `G15` | `02` (PDF, Luna) | `MS5248M` | `Болюс: Нет.` | `mismatch` (`true`) | `mismatch`, `cos=0.816` (`true`) | `mismatch` |

---

## 5. Результаты запуска `H2` (`ai-forever/FRIDA` + `hybrid_rrf`) на проверенной выборке

После подключения энкодера семантических слотов в [`embeddings.py`](file:///Users/user/Developer/agentic_tender/src/tenderai/embeddings.py) и слияния рангов `RRF (k=60)` в [`align.py`](file:///Users/user/Developer/agentic_tender/src/tenderai/align.py) (подробные цифры — в [H2.json](file:///Users/user/Developer/agentic_tender/runs/experiments/H2.json)):

- **Точность выравнивания характеристик (`alignment_accuracy`) выросла с `10/15 (0.667)` до `14/15 (0.933)`** (`+26.7` п.п.).
- **`Recall@1` поднялся с `0.833` до `0.917` (`11/11` на одиночных слотах), а `Recall@3` достиг `1.000` (`12/12`)** — даже для составного требования `G14` обе нужные строки паспорта `M275-07` (`Режим остановки «0»` и `Рабочие скорости без режима «0»`) стоят на 1-м и 3-м местах в выдаче.
- **Ложные `match` полностью устранены (`1 → 0`)**: `G11` переведён из ложного `match` в честный `unknown`, `G12` восстановлен из `not_aligned` в подтверждённый `match`, а отсутствующие характеристики `G08` и `G13` чисто отсечены порогом близости `0.625` (`absent_abstention_accuracy = 3/3`).
- **`macro-F1` вырос с `0.869` до `0.937`, а доказательная точность `match` (`verified_match_precision`) — с `0.714` до `1.000` (`6/6`)**.
- Полная типографская версия всех таблиц и разборов включена в [docs/final_report.tex](file:///Users/user/Developer/agentic_tender/docs/final_report.tex) ([docs/final_report.pdf](file:///Users/user/Developer/agentic_tender/docs/final_report.pdf)).
