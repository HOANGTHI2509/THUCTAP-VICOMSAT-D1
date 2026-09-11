# VCOMSAT ΓÇö Real-time Fuel Data Denoising & Filtering (─Éß╗ü t├ái 1)

> **Hß╗ç thß╗æng xß╗¡ l├╜ nhiß╗àu v├á lß╗ìc t├¡n hiß╗çu mß╗⌐c nhi├¬n liß╗çu viß╗àn th├┤ng theo thß╗¥i gian thß╗▒c (Causal Filtering)**  
> Phi├¬n bß║ún b├án giao: `1.2.0-enterprise` | Trß║íng th├íi kiß╗âm thß╗¡: `83/83 unit/regression tests passed`, `18/19 golden behavior checks`

---

## 1. Giß╗¢i thiß╗çu dß╗▒ ├ín

### 1.1. Bß╗æi cß║únh v├á B├ái to├ín kß╗╣ thuß║¡t
Trong c├íc hß╗ç thß╗æng gi├ím s├ít h├ánh tr├¼nh ph╞░╞íng tiß╗çn vß║¡n tß║úi (FMS / Telematics), cß║úm biß║┐n mß╗⌐c nhi├¬n liß╗çu lß║»p ─æß║╖t trong b├¼nh dß║ºu li├¬n tß╗Ñc gß╗¡i dß╗» liß╗çu vß╗ü trung t├óm ─æiß╗üu h├ánh. Tuy nhi├¬n, t├¡n hiß╗çu ─æo th├┤ (**RawFuel**) lu├┤n bß╗ï ├┤ nhiß╗àm bß╗ƒi c├íc loß║íi nhiß╗àu vß║¡t l├╜ phß╗⌐c tß║íp:
- **Nhiß╗àu s├│ng s├ính (Sloshing) & Rung lß║»c**: Nhi├¬n liß╗çu va ─æß║¡p v├áo th├ánh b├¼nh khi xe t─âng tß╗æc, phanh gß║Ñp, v├áo cua hoß║╖c di chuyß╗ân tr├¬n ─æ╞░ß╗¥ng gß╗ô ghß╗ü.
- **Biß║┐n dß║íng do ─æß╗Ö dß╗æc / ─Éß╗ïa h├¼nh**: Xe leo dß╗æc hoß║╖c xuß╗æng dß╗æc l├ám phao cß║úm biß║┐n nghi├¬ng, tß║ío ra c├íc b╞░ß╗¢c nhß║úy mß╗⌐c giß║ú tß║ím thß╗¥i dß║íng chß╗» U hoß║╖c dß║íng ─æß╗ôi.
- **Lß╗ùi phß║ºn cß╗⌐ng cß║úm biß║┐n & Mß║Ñt nguß╗ôn**: T├¡n hiß╗çu rß╗¢t ─æß╗Öt ngß╗Öt vß╗ü `0.0 L` trong 1ΓÇô2 chu kß╗│ (Zero Dropout), hoß║╖c xung ─æiß╗çn ├íp g├óy vß╗ìt ─æß╗ënh cß╗▒c ─æß║íi (Spike).
- **Tr├┤i t├¡n hiß╗çu & Nhiß╗àu dß╗½ng**: Khi xe ─æß╗ù nß╗ò m├íy hoß║╖c tß║»t m├íy, cß║úm biß║┐n vß║½n dao ─æß╗Öng nhß║╣ quanh mß║╖t bß║▒ng thß╗▒c tß║┐ do nhiß╗àu nhiß╗çt v├á rung ─æß╗Öng c╞í (Stable Jitter).
- **Sai sß╗æ ─æß╗ïnh vß╗ï GPS**: Vß║¡n tß╗æc b├ío 0 km/h nh╞░ng toß║í ─æß╗Ö GPS nhß║úy do hiß╗çu ß╗⌐ng phß║ún xß║í ─æa ─æ╞░ß╗¥ng (Multipath), g├óy kh├│ kh─ân cho viß╗çc ph├ón ─æß╗ïnh trß║íng th├íi xe.

### 1.2. Mß╗Ñc ti├¬u hß╗ç thß╗æng
X├óy dß╗▒ng mß╗Öt dß╗ïch vß╗Ñ lß╗ìc dß╗» liß╗çu thß╗¥i gian thß╗▒c ─æß╗Öc lß║¡p (**Real-time Streaming Microservice**), tiß║┐p nhß║¡n luß╗ông dß╗» liß╗çu ─æo th├┤ ─æ├ú quy ─æß╗òi sang l├¡t, b├│c t├ích to├án bß╗Ö c├íc dß║íng nhiß╗àu v├á tr├¡ch xuß║Ñt ─æ╞░ß╗¥ng nhi├¬n liß╗çu thß╗▒c sß╗▒ phß║ún ├ính mß╗⌐c ti├¬u hao v├á mß║╖t bß║▒ng thß╗▒c tß║┐ (**CleanFuel - ─É╞░ß╗¥ng m├áu t├¡m**).

![So s├ính trß╗▒c quan hiß╗çu quß║ú khß╗¡ nhiß╗àu](docs/images/filter_comparison_visual.png)
*H├¼nh 1: ─Éß╗æi s├ính giß╗»a t├¡n hiß╗çu ─æo th├┤ (RawFuel) v├á c├íc ph╞░╞íng ph├íp lß╗ìc, l├ám nß╗òi bß║¡t ─æ╞░ß╗¥ng lß╗ìc m├áu t├¡m th├¡ch ß╗⌐ng giß╗» ß╗òn ─æß╗ïnh khi c├│ rung lß║»c.*

```text
V├¡ dß╗Ñ khß╗¡ nhiß╗àu thß╗▒c tß║┐ (Xe dß╗½ng nß╗ò m├íy, cß║úm biß║┐n rung lß║»c):
RawFuel (Th├┤):     300.0 L ΓöÇΓöÇ> 280.0 L ΓöÇΓöÇ> 279.0 L ΓöÇΓöÇ> 281.0 L ΓöÇΓöÇ> 300.0 L
CleanFuel (T├¡m):   300.0 L ΓöÇΓöÇ> 300.0 L ΓöÇΓöÇ> 300.0 L ΓöÇΓöÇ> 300.0 L ΓöÇΓöÇ> 300.0 L
QualityFlag:       INITIAL ΓöÇΓöÇ> VALLEY_HOLD ΓöÇΓöÇ> VALLEY_HOLD ΓöÇΓöÇ> VALLEY_HOLD ΓöÇΓöÇ> RECOVERY_SMOOTH
```

### 1.3. Phß║ím vi nghiß╗çm thu ─Éß╗ü t├ái 1
- **Nhiß╗çm vß╗Ñ trß╗ìng t├óm**: Tiß║┐p nhß║¡n telemetry thß╗¥i gian thß╗▒c, khß╗¡ nhiß╗àu, l├ám m╞░ß╗út th├¡ch ß╗⌐ng v├á ─æ├ính gi├í ─æß╗Ö tin cß║¡y t├¡n hiß╗çu (`CleanFuel`, `SignalState`, `QualityFlag`, `MotionState`).
- **Giß╗¢i hß║ín nghiß╗çp vß╗Ñ (Ranh giß╗¢i ─Éß╗ü t├ái 1 v├á ─Éß╗ü t├ái 2)**:
  - Hß╗ç thß╗æng **KH├öNG** ─æ╞░a ra kß║┐t luß║¡n nghiß╗çp vß╗Ñ nh╞░ "Xe vß╗½a nß║íp nhi├¬n liß╗çu" hay "Xe bß╗ï r├║t trß╗Öm nhi├¬n liß╗çu".
  - C├íc trß║íng th├íi nh├ún nh╞░ `UPWARD_SHIFT` hay `DOWNWARD_SHIFT` chß╗ë thuß║ºn t├║y m├┤ tß║ú h├¼nh hß╗ìc t├¡n hiß╗çu (mß║╖t bß║▒ng ─æo ─æ╞░ß╗úc dß╗ïch chuyß╗ân t─âng hoß║╖c giß║úm bß╗ün vß╗»ng). Quyß║┐t ─æß╗ïnh sß╗▒ kiß╗çn nß║íp/h├║t thuß╗Öc vß╗ü tß║ºng ph├ón t├¡ch nghiß╗çp vß╗Ñ ph├¡a sau (─Éß╗ü t├ái 2).
  - Hß╗ç thß╗æng **KH├öNG** y├¬u cß║ºu t├¡n hiß╗çu ch├¼a kh├│a (ACC/Ignition) hay cß║úm biß║┐n ─æß╗Ö cao/─æß╗Ö nghi├¬ng (v├¼ phß║ºn cß╗⌐ng hiß╗çn tß║íi cß╗ºa kh├ích h├áng ch╞░a trang bß╗ï).

---

## 2. Kiß║┐n tr├║c tß╗òng thß╗â

Hß╗ç thß╗æng hoß║ít ─æß╗Öng theo nguy├¬n l├╜ **Causal Stream Processing**: mß╗ùi ─æiß╗âm dß╗» liß╗çu ─æß║┐n ─æ╞░ß╗úc xß╗¡ l├╜ ngay lß║¡p tß╗⌐c chß╗ë dß╗▒a v├áo gi├í trß╗ï hiß╗çn tß║íi v├á l╞░ß╗úc sß╗¡ qu├í khß╗⌐ cß╗ºa ch├¡nh ph╞░╞íng tiß╗çn ─æ├│.

### 2.1. S╞í ─æß╗ô luß╗ông dß╗» liß╗çu (Architecture Pipeline)

```mermaid
flowchart LR
    A["Thiß║┐t bß╗ï GPS & Cß║úm biß║┐n tr├¬n xe"] --> B["API Telemetry Gateway (/clean-point)"]
    B --> C["Kiß╗âm tra & Chuß║⌐n h├│a dß╗» liß╗çu (Sanitization)"]
    C --> D["Tr├¡ch xuß║Ñt ─æß║╖c tr╞░ng Causal (Features Engine)"]
    D --> E["M├┤ h├¼nh AI ph├ón loß║íi SignalState"]
    D --> I["─É├ính gi├í vß║¡n ─æß╗Öng MotionState (Speed + GPS)"]
    E --> F["L├╡i lß╗ìc th├¡ch ß╗⌐ng AI Smooth-Tracking (Adaptive Kalman)"]
    I --> F
    F --> G["CleanFuel (─É╞░ß╗¥ng t├¡m)"]
    F --> H["QualityFlag (H├ánh ─æß╗Öng lß╗ìc)"]
    G --> J["L╞░u CSDL / Microservice nghiß╗çp vß╗Ñ ─Éß╗ü t├ái 2"]
    H --> J
```

### 2.2. Chi tiß║┐t chß╗⌐c n─âng 8 tß║ºng xß╗¡ l├╜
1. **Tß║ºng tiß║┐p nhß║¡n (Ingestion Gateway)**: Nhß║¡n bß║ún tin JSON qua REST API (`/api/v1/fuel/clean-point` hoß║╖c `/clean-batch`), kiß╗âm tra API Key v├á ─æß║⌐y v├áo h├áng ─æß╗úi ─æ╞ín luß╗ông theo tß╗½ng xe (`VehicleQueueManager`).
2. **Tß║ºng chuß║⌐n h├│a (Sanitization & Validation)**: Kiß╗âm tra ─æß╗ïnh dß║íng thß╗¥i gian ISO-8601, loß║íi bß╗Å tß╗ìa ─æß╗Ö GPS kh├┤ng hß╗úp lß╗ç (nh╞░ `0, 0`), ph├ít hiß╗çn gi├í trß╗ï ├óm, rß╗¢t vß╗ü 0 hoß║╖c v╞░ß╗út trß║ºn dung t├¡ch b├¼nh (`CapacityEst`).
3. **Tß║ºng x├íc ─æß╗ïnh vß║¡n ─æß╗Öng (Motion Assessment)**: Kß║┐t hß╗úp vß║¡n tß╗æc tß╗⌐c thß╗¥i v├á b├ín k├¡nh dß╗ïch chuyß╗ân GPS trong cß╗¡a sß╗ò tr╞░ß╗út 5 ─æiß╗âm gß║ºn nhß║Ñt ─æß╗â g├ín nh├ún trß║íng th├íi vß║¡n ─æß╗Öng (`MOVING`, `LOW_MOTION`, `UNCERTAIN`).
4. **Tß║ºng tr├¡ch xuß║Ñt ─æß║╖c tr╞░ng (Causal Feature Extraction)**: T├¡nh to├ín ─æß╗Ö biß║┐n thi├¬n, ─æß╗Ö lß╗çch chuß║⌐n tr╞░ß╗út, h╞░ß╗¢ng dß╗æc v├á ─æiß╗âm ph├ón kß╗│ chß╗ë tß╗½ dß╗» liß╗çu qu├í khß╗⌐.
5. **Tß║ºng ph├ón loß║íi t├¡n hiß╗çu AI (AI Signal State Classifier)**: M├┤ h├¼nh Random Forest sß╗¡ dß╗Ñng vector ─æß║╖c tr╞░ng ─æß╗â ph├ón loß║íi dß║íng t├¡n hiß╗çu th├ánh 5 trß║íng th├íi chuß║⌐n (`STABLE_JITTER`, `GRADUAL_CHANGE`, `OSCILLATION_NOISE`, `UPWARD_SHIFT`, `DOWNWARD_SHIFT`).
6. **Tß║ºng lß╗ìc th├¡ch ß╗⌐ng AI Smooth-Tracking (Adaptive Kalman Core)**: ─Éiß╗üu chß╗ënh ─æß╗Öng hiß╗çp ph╞░╞íng sai nhiß╗àu ─æo $R$ v├á nhiß╗àu hß╗ç thß╗æng $Q$ dß╗▒a tr├¬n kß║┐t hß╗úp giß╗»a `SignalState`, `MotionState` v├á dung t├¡ch xe.
7. **Tß║ºng quß║ún l├╜ trß║íng th├íi xe (Vehicle State Store)**: ─É├│ng g├│i v├á l╞░u vß║┐t State Context cß╗ºa xe (Kalman state, lß╗ïch sß╗¡ ─æß╗çm, bß╗Ö ─æß║┐m x├íc nhß║¡n) v├áo RAM hoß║╖c Redis (c├│ kh├│a an to├án chß╗æng Race Condition).
8. **Tß║ºng xuß║Ñt dß╗» liß╗çu (Contract Delivery)**: Trß║ú vß╗ü kß║┐t quß║ú JSON ─æß╗ông nhß║Ñt bao gß╗ôm gi├í trß╗ï sß║ích, cß╗¥ chß║Ñt l╞░ß╗úng v├á ─æß╗Ö trß╗à t├¡nh to├ín (Latency).

---

## 3. Cß║Ñu tr├║c th╞░ mß╗Ñc m├ú nguß╗ôn

Hß╗ç thß╗æng ─æ╞░ß╗úc module h├│a chß║╖t chß║╜, ph├ón t├ích r├╡ giß╗»a thuß║¡t to├ín l├╡i, tß║ºng dß╗ïch vß╗Ñ, c├┤ng cß╗Ñ kiß╗âm thß╗¡ v├á t├ái liß╗çu:

```text
THUCTAP-VCOMSAT/
Γö£ΓöÇΓöÇ src/
Γöé   Γö£ΓöÇΓöÇ core/                                # Tß║ºng xß╗¡ l├╜ t├¡n hiß╗çu l├╡i
Γöé   Γöé   ΓööΓöÇΓöÇ filters/                         # C├íc thuß║¡t to├ín lß╗ìc t├¡n hiß╗çu nhi├¬n liß╗çu
Γöé   Γöé       Γö£ΓöÇΓöÇ smooth_tracking/             # Thuß║¡t to├ín lß╗ìc m╞░ß╗út th├¡ch ß╗⌐ng thß╗¥i gian thß╗▒c (AI Smooth-Tracking)
Γöé   Γöé       Γöé   Γö£ΓöÇΓöÇ __init__.py              # Khß╗ƒi tß║ío package v├á xuß║Ñt interface chuß║⌐n
Γöé   Γöé       Γöé   Γö£ΓöÇΓöÇ config.py                # Cß║Ñu h├¼nh tham sß╗æ lß╗ìc (ma trß║¡n Q, R, ng╞░ß╗íng dß╗ïch chuyß╗ân)
Γöé   Γöé       Γöé   Γö£ΓöÇΓöÇ contracts.py             # ─Éß╗ïnh ngh─⌐a cß║Ñu tr├║c dß╗» liß╗çu v├á chuß║⌐n h├│a trß║íng th├íi
Γöé   Γöé       Γöé   Γö£ΓöÇΓöÇ dataframe.py             # Bß╗Ö ─æiß╗üu phß╗æi xß╗¡ l├╜ theo l├┤ (batch processing) cho DataFrame
Γöé   Γöé       Γöé   Γö£ΓöÇΓöÇ engine.py                # ─Éß╗Öng c╞í ─æiß╗üu phß╗æi lß╗ìc trß╗▒c tuyß║┐n theo tß╗½ng ─æiß╗âm ─æo
Γöé   Γöé       Γöé   Γö£ΓöÇΓöÇ features.py              # Tr├¡ch xuß║Ñt ─æß║╖c tr╞░ng nh├ón quß║ú (Causal Features)
Γöé   Γöé       Γöé   Γö£ΓöÇΓöÇ kalman.py                # Thuß║¡t to├ín lß╗ìc Kalman th├¡ch ß╗⌐ng 1D
Γöé   Γöé       Γöé   ΓööΓöÇΓöÇ state.py                 # Quß║ún l├╜ v├á l╞░u trß╗» ngß╗» cß║únh trß║íng th├íi theo tß╗½ng xe
Γöé   Γöé       Γö£ΓöÇΓöÇ ai_smooth_tracking_filter.py       # Facade t╞░╞íng th├¡ch ng╞░ß╗úc cho c├íc module c┼⌐
Γöé   Γöé       ΓööΓöÇΓöÇ ai_enhanced_adaptive_realtime.py   # Thuß║¡t to├ín Adaptive Kalman Filter 1D ─æß╗æi chß╗⌐ng
Γöé   Γö£ΓöÇΓöÇ service/                             # Tß║ºng dß╗ïch vß╗Ñ Microservice REST API & Quß║ún l├╜ State
Γöé   Γöé   Γö£ΓöÇΓöÇ api.py                           # REST API endpoint thß╗¥i gian thß╗▒c (FastAPI)
Γöé   Γöé   Γö£ΓöÇΓöÇ queue_manager.py                 # Quß║ún l├╜ h├áng ─æß╗úi FIFO tuß║ºn tß╗▒ theo tß╗½ng xe
Γöé   Γöé   Γö£ΓöÇΓöÇ state_manager.py                 # ─Éiß╗üu phß╗æi l╞░u trß╗» trß║íng th├íi ph╞░╞íng tiß╗çn
Γöé   Γöé   ΓööΓöÇΓöÇ state_store.py                   # Tß║ºng trß╗½u t╞░ß╗úng h├│a bß╗Ö nhß╗¢ State (Memory / Redis)
Γöé   Γö£ΓöÇΓöÇ sdk/
Γöé   Γöé   ΓööΓöÇΓöÇ fuel_cleaner.py                  # Th╞░ viß╗çn Python SDK t├¡ch hß╗úp trß╗▒c tiß║┐p kh├┤ng qua mß║íng
Γöé   Γö£ΓöÇΓöÇ dashboard/
Γöé   Γöé   Γö£ΓöÇΓöÇ app_dashboard_tienxuly.py        # Giao diß╗çn trß╗▒c quan h├│a v├á gi├ím s├ít t├¡n hiß╗çu (Streamlit)
Γöé   Γöé   ΓööΓöÇΓöÇ dashboard_data.py                # Module nß║íp v├á chuß║⌐n h├│a dß╗» liß╗çu viß╗àn th├┤ng ─æa nguß╗ôn
Γöé   ΓööΓöÇΓöÇ pipeline/
Γöé       ΓööΓöÇΓöÇ train_fuel_state_classifier.py   # Quy tr├¼nh huß║Ñn luyß╗çn m├┤ h├¼nh Machine Learning offline
Γö£ΓöÇΓöÇ models/                                  # Trß╗ìng sß╗æ m├┤ h├¼nh Machine Learning
Γöé   ΓööΓöÇΓöÇ fuel_state_classifier/               # M├┤ h├¼nh Random Forest 28 ─æß║╖c tr╞░ng
Γöé       Γö£ΓöÇΓöÇ fuel_state_classifier.pkl        # File trß╗ìng sß╗æ m├┤ h├¼nh ─æ├ú huß║Ñn luyß╗çn
Γöé       Γö£ΓöÇΓöÇ metadata.json                    # Danh s├ích 28 ─æß║╖c tr╞░ng v├á si├¬u tham sß╗æ
Γöé       ΓööΓöÇΓöÇ test_confusion_matrix.png        # Ma trß║¡n nhß║ºm lß║½n gß╗æc tr├¬n tß║¡p kiß╗âm thß╗¡
Γö£ΓöÇΓöÇ reports/
Γöé   ΓööΓöÇΓöÇ confusion_matrices/                  # B├ío c├ío ─æ├ính gi├í ma trß║¡n nhß║ºm lß║½n 34 ph╞░╞íng tiß╗çn
Γöé       Γö£ΓöÇΓöÇ summary_per_vehicle.md           # B├ío c├ío chi tiß║┐t dß║íng v─ân bß║ún Markdown
Γöé       Γö£ΓöÇΓöÇ fleet_accuracy_summary.csv       # Tß╗òng hß╗úp ph├ón bß╗æ mß║½u v├á ─æß╗Ö ch├¡nh x├íc to├án hß║ím ─æß╗Öi
Γöé       Γö£ΓöÇΓöÇ svg/                             # Biß╗âu ─æß╗ô vector SVG tß╗½ng xe (cm_<xe>.svg)
Γöé       ΓööΓöÇΓöÇ csv/                             # Bß║úng ma trß║¡n nhß║ºm lß║½n dß║íng CSV tß╗½ng xe
Γö£ΓöÇΓöÇ docs/                                    # T├ái liß╗çu thiß║┐t kß║┐ kß╗╣ thuß║¡t v├á t├ái nguy├¬n ─æß╗ô hß╗ìa
Γöé   Γö£ΓöÇΓöÇ images/                              # Biß╗âu ─æß╗ô kß╗╣ thuß║¡t v├á ma trß║¡n dß║íng vector SVG / PNG
Γöé   Γöé   Γö£ΓöÇΓöÇ cm_Car_5.svg                     # Ma trß║¡n nhß║ºm lß║½n xe ─æß║íi diß╗çn Car 5
Γöé   Γöé   Γö£ΓöÇΓöÇ test_held_out_confusion_matrix.svg # Ma trß║¡n nhß║ºm lß║½n tr├¬n tß║¡p Test ─æß╗Öc lß║¡p
Γöé   Γöé   Γö£ΓöÇΓöÇ overall_fleet_confusion_matrix.svg # Ma trß║¡n nhß║ºm lß║½n tß╗òng hß╗úp to├án bß╗Ö 34 xe
Γöé   Γöé   Γö£ΓöÇΓöÇ filter_comparison_visual.png     # Biß╗âu ─æß╗ô so s├ính trß╗▒c quan c├íc ph╞░╞íng ph├íp lß╗ìc
Γöé   Γöé   ΓööΓöÇΓöÇ adaptive_kalman_behavior.png     # Biß╗âu ─æß╗ô c╞í chß║┐ b├ím th├¡ch ß╗⌐ng cß╗ºa Kalman
Γöé   ΓööΓöÇΓöÇ API_DOCUMENTATION.md                 # T├ái liß╗çu ─æß║╖c tß║ú kß╗╣ thuß║¡t REST API
Γö£ΓöÇΓöÇ tests/                                   # Bß╗Ö kiß╗âm thß╗¡ tß╗▒ ─æß╗Öng to├án diß╗çn (83 tests)
Γöé   Γö£ΓöÇΓöÇ fixtures/
Γöé   Γöé   Γö£ΓöÇΓöÇ golden_fuel_segments.json        # Dß╗» liß╗çu kiß╗âm thß╗¡ chuß║⌐n tß╗½ c├íc ─æoß║ín vß║¡n h├ánh thß╗▒c tß║┐
Γöé   Γöé   ΓööΓöÇΓöÇ README.md                        # H╞░ß╗¢ng dß║½n quy tr├¼nh ─æ├ính gi├í v├á nghiß╗çm thu dß╗» liß╗çu chuß║⌐n
Γöé   Γö£ΓöÇΓöÇ test_real_data_golden_segments.py    # Kiß╗âm thß╗¡ hß╗ôi quy tr├¬n c├íc ─æoß║ín dß╗» liß╗çu thß╗▒c tß║┐
Γöé   Γö£ΓöÇΓöÇ test_concurrent_streaming.py         # Kiß╗âm thß╗¡ an to├án luß╗ông v├á xß╗¡ l├╜ tuß║ºn tß╗▒ FIFO
Γöé   Γö£ΓöÇΓöÇ test_motion_quality_context.py       # Kiß╗âm thß╗¡ logic ph├ón ─æß╗ïnh trß║íng th├íi vß║¡n tß╗æc v├á GPS
Γöé   ΓööΓöÇΓöÇ test_purple_service_unification.py   # Kiß╗âm thß╗¡ t├¡nh nhß║Ñt qu├ín giß╗»a API, SDK v├á Core Engine
Γö£ΓöÇΓöÇ scripts/
Γöé   Γö£ΓöÇΓöÇ generate_per_vehicle_confusion_matrix.py # Script sinh ma trß║¡n nhß║ºm lß║½n cho 34 ph╞░╞íng tiß╗çn
Γöé   Γö£ΓöÇΓöÇ export_confusion_matrix_svg.py       # Script xuß║Ñt ─æß╗ô hß╗ìa vector SVG chß║Ñt l╞░ß╗úng cao
Γöé   Γö£ΓöÇΓöÇ evaluate_smooth_tracking.py          # Script ─æ├ính gi├í ─æß╗ïnh l╞░ß╗úng KPI bß╗Ö lß╗ìc
Γöé   ΓööΓöÇΓöÇ find_golden_candidates.py            # C├┤ng cß╗Ñ tr├¡ch xuß║Ñt ─æoß║ín t├¡n hiß╗çu mß║½u tß╗½ dß╗» liß╗çu th├┤
Γö£ΓöÇΓöÇ artifacts/
Γöé   ΓööΓöÇΓöÇ evaluation/                          # Kß║┐t quß║ú ─æo l╞░ß╗¥ng KPI, metrics.json v├á b├ío c├ío hiß╗çu n─âng
Γö£ΓöÇΓöÇ Dockerfile                               # Cß║Ñu h├¼nh container ─æ├│ng g├│i Microservice
Γö£ΓöÇΓöÇ docker-compose.yml                       # File ─æiß╗üu phß╗æi khß╗ƒi chß║íy hß╗ç thß╗æng k├¿m Redis
ΓööΓöÇΓöÇ requirements.txt                         # Danh s├ích th╞░ viß╗çn v├á g├│i phß╗Ñ thuß╗Öc
```

> **L╞░u ├╜ quan trß╗ìng cho kß╗╣ s╞░ t├¡ch hß╗úp**:
> - **Entry point dß╗ïch vß╗Ñ mß║íng**: [src/service/api.py](file:///d:/THUCTAP_VICOMSAT/src/service/api.py).
> - **Entry point thuß║¡t to├ín ─æ╞░ß╗¥ng t├¡m**: [src/core/filters/smooth_tracking/engine.py](file:///d:/THUCTAP_VICOMSAT/src/core/filters/smooth_tracking/engine.py) (`SmoothTrackingFilterEngine`).
> - **N╞íi l╞░u to├án bß╗Ö tham sß╗æ kß╗╣ thuß║¡t**: [src/core/filters/smooth_tracking/config.py](file:///d:/THUCTAP_VICOMSAT/src/core/filters/smooth_tracking/config.py).

---

## 4. Dß╗» liß╗çu ─æß║ºu v├áo (Input Contract)

Hß╗ç thß╗æng xß╗¡ l├╜ tß╗½ng ─æiß╗âm ─æo ─æß╗Öc lß║¡p theo luß╗ông JSON gß╗¡i l├¬n API.

### 4.1. Tß╗½ ─æiß╗ân dß╗» liß╗çu (Data Dictionary)

| T├¬n tr╞░ß╗¥ng | Kiß╗âu dß╗» liß╗çu | ─É╞ín vß╗ï | Bß║»t buß╗Öc | ├¥ ngh─⌐a kß╗╣ thuß║¡t |
| :--- | :--- | :--- | :---: | :--- |
| `VehicleID` | `string` | ΓÇö | **C├│** | M├ú ─æß╗ïnh danh duy nhß║Ñt cß╗ºa xe (Biß╗ân sß╗æ xe hoß║╖c ID thiß║┐t bß╗ï). |
| `FuelTime` | `datetime` | ISO-8601 | **C├│** | Mß╗æc thß╗¥i gian ghi nhß║¡n (V├¡ dß╗Ñ: `2026-08-27T10:00:00`). |
| `FuelLevel` | `float` | **L├¡t** | **C├│** | Mß╗⌐c nhi├¬n liß╗çu th├┤ ─æo tß╗½ cß║úm biß║┐n, **bß║»t buß╗Öc ─æ├ú quy ─æß╗òi sang l├¡t**. |
| `Speed` | `float` | km/h | Kh├┤ng | Vß║¡n tß╗æc tß╗⌐c thß╗¥i tß╗½ GPS (Mß║╖c ─æß╗ïnh: `0.0`). |
| `Lat` | `float` | ─Éß╗Ö thß║¡p ph├ón | Kh├┤ng | V─⌐ ─æß╗Ö GPS (V├¡ dß╗Ñ: `21.0285`). Bß╗Å qua nß║┐u lß╗ùi. |
| `Lng` | `float` | ─Éß╗Ö thß║¡p ph├ón | Kh├┤ng | Kinh ─æß╗Ö GPS (V├¡ dß╗Ñ: `105.8542`). Bß╗Å qua nß║┐u lß╗ùi. |
| `CapacityEst` | `float` | L├¡t | Kh├┤ng | Dung t├¡ch b├¼nh nhi├¬n liß╗çu ╞░ß╗¢c t├¡nh (Mß║╖c ─æß╗ïnh: `200.0 L`). |

### 4.2. C├íc quy ╞░ß╗¢c bß║»t buß╗Öc khi vß║¡n h├ánh
1. **─É╞ín vß╗ï chuß║⌐n h├│a**: Dß╗» liß╗çu cß║úm biß║┐n truyß╗ün v├áo phß║úi ─æ╞░ß╗úc t├¡nh bß║▒ng **L├¡t**. Hß╗ç thß╗æng kh├┤ng nhß║¡n gi├í trß╗ï ADC/Volt th├┤ ch╞░a qua bß║úng hiß╗çu chuß║⌐n (Calib).
2. **Thß╗⌐ tß╗▒ thß╗¥i gian (Chronological Order)**: C├íc ─æiß╗âm ─æo cß╗ºa c├╣ng mß╗Öt `VehicleID` phß║úi ─æ╞░ß╗úc gß╗¡i ─æß║┐n theo ─æ├║ng thß╗⌐ tß╗▒ thß╗¥i gian t─âng dß║ºn (`FuelTime[t] >= FuelTime[t-1]`). Nß║┐u ─æiß╗âm gß╗¡i tß╗¢i c├│ thß╗¥i gian c┼⌐ h╞ín ─æiß╗âm cuß╗æi ─æ├ú xß╗¡ l├╜, hß╗ç thß╗æng sß║╜ ─æ├ính dß║Ñu `ORDER_VIOLATION` v├á giß╗» nguy├¬n mß╗⌐c nhi├¬n liß╗çu sß║ích.
3. **Ph├ón lß║¡p trß║íng th├íi theo xe**: Mß╗ùi `VehicleID` sß╗ƒ hß╗»u mß╗Öt State Context ─æß╗Öc lß║¡p ho├án to├án. Dß╗» liß╗çu cß╗ºa xe 29E-45520 tuyß╗çt ─æß╗æi kh├┤ng ß║únh h╞░ß╗ƒng tß╗¢i trß║íng th├íi lß╗ìc cß╗ºa xe 21H-02058.
4. **Tß╗ìa ─æß╗Ö GPS kh├┤ng hß╗úp lß╗ç**: Cß║╖p tß╗ìa ─æß╗Ö `(0.0, 0.0)` hoß║╖c tß╗ìa ─æß╗Ö ngo├ái dß║úi ─æß╗ïa l├╜ Viß╗çt Nam ─æ╞░ß╗úc xem l├á lß╗ùi vß╗ç tinh v├á bß╗ï loß║íi bß╗Å khß╗Åi t├¡nh to├ín cß╗▒ ly.
5. **C╞í chß║┐ tß╗▒ suy luß║¡n `CapacityEst`**:
   - Nß║┐u doanh nghiß╗çp truyß╗ün `CapacityEst`, hß╗ç thß╗æng sß║╜ sß╗¡ dß╗Ñng gi├í trß╗ï n├áy ─æß╗â ─æß╗ïnh tß╗╖ lß╗ç c├íc ng╞░ß╗íng lß╗ìc (Spike, Jitter, Sloshing).
   - Nß║┐u kh├┤ng truyß╗ün hoß║╖c truyß╗ün gi├í trß╗ï `<= 30.0 L`, hß╗ç thß╗æng sß║╜ tß╗▒ suy luß║¡n tß║ím thß╗¥i tß╗½ ─æiß╗âm nhi├¬n liß╗çu hß╗úp lß╗ç ─æß║ºu ti├¬n: `CapacityEst = RawFuel * 1.05` (tß╗æi thiß╗âu `200.0 L`). Khi ph├ít hiß╗çn `RawFuel` v╞░ß╗út dung t├¡ch tß║ím, hß╗ç thß╗æng tß╗▒ ─æß╗Öng co gi├ún ng╞░ß╗íng l├¬n ─æß╗â th├¡ch ß╗⌐ng.

---

## 5. Quy tr├¼nh tiß╗ün xß╗¡ l├╜ dß╗» liß╗çu (Sanitization & Segmentation)

### 5.1. Chuß║⌐n h├│a dß╗» liß╗çu ─æß║ºu v├áo (Sanitization)
- **Parse thß╗¥i gian chß║╖t chß║╜**: To├án bß╗Ö chuß╗ùi thß╗¥i gian ─æ╞░ß╗úc chuß║⌐n h├│a vß╗ü ─æß╗ïnh dß║íng ISO-8601. C├íc bß║ún ghi sai ─æß╗ïnh dß║íng hoß║╖c mß╗æc thß╗¥i gian kh├┤ng hß╗úp lß╗ç bß╗ï tß╗½ chß╗æi ngay tß║íi tß║ºng API.
- **├ëp kiß╗âu an to├án**: Chuyß╗ân ─æß╗òi c├íc gi├í trß╗ï sß╗æ thß╗▒c (`float`), loß║íi bß╗Å chuß╗ùi r├íc.
- **Kh├┤ng nß╗Öi suy t╞░╞íng lai trong chß║┐ ─æß╗Ö Real-time**: Tuyß╗çt ─æß╗æi kh├┤ng sß╗¡ dß╗Ñng c├íc kß╗╣ thuß║¡t nh╞░ Spline nß╗Öi suy hay Rolling Center Window vß╗æn ─æ├▓i hß╗Åi biß║┐t tr╞░ß╗¢c dß╗» liß╗çu t╞░╞íng lai. Mß╗ìi ph├⌐p xß╗¡ l├╜ chß╗ë ─æ╞░ß╗úc nh├¼n thß║Ñy $t \le t_{hiß╗çn\_tß║íi}$.

### 5.2. Kiß╗âm tra t├¡nh hß╗úp lß╗ç v├á xß╗¡ l├╜ bi├¬n (Validation Rules)
1. **Raw bß║▒ng 0 (`RawFuel == 0.0`)**: ─É╞░ß╗úc nhß║¡n diß╗çn l├á mß║Ñt t├¡n hiß╗çu nguß╗ôn cß║úm biß║┐n (Zero Dropout). Hß╗ç thß╗æng k├¡ch hoß║ít trß║íng th├íi giß╗» nguy├¬n mß╗⌐c sß║ích tr╞░ß╗¢c ─æ├│ (`DROPOUT_ZERO_HOLD`).
2. **Raw ├óm (`RawFuel < 0.0`) hoß║╖c gi├í trß╗ï `NaN`**: ─É╞░ß╗úc xem l├á lß╗ùi dß╗» liß╗çu ─æ╞░ß╗¥ng truyß╗ün. Bß╗Ö lß╗ìc bß╗Å qua gi├í trß╗ï ─æo n├áy v├á duy tr├¼ mß╗⌐c c┼⌐ vß╗¢i cß║únh b├ío `INVALID_INPUT`.
3. **Raw v╞░ß╗út giß╗¢i hß║ín vß║¡t l├╜ (`RawFuel > CapacityEst * 1.02`)**: Xß╗¡ l├╜ nh╞░ gi├í trß╗ï cß╗▒c ─æß║íi bß║Ñt th╞░ß╗¥ng, kh├┤ng cß║¡p nhß║¡t Kalman trß╗▒c tiß║┐p m├á ─æ╞░a v├áo nh├ính kiß╗âm tra ─æß╗Öt biß║┐n.
4. **Bß║ún tin tß╗¢i trß╗à / Sai thß╗⌐ tß╗▒**: ─Éiß╗âm ─æo c├│ thß╗¥i gian nhß╗Å h╞ín thß╗¥i gian gß║ºn nhß║Ñt cß╗ºa xe sß║╜ bß╗ï bß╗Å qua v├á giß╗» nguy├¬n mß╗⌐c Clean.
5. **GPS nhß║úy c├│c (GPS Glitch)**: Khi vß║¡n tß╗æc xe b├ío `0.0 km/h` nh╞░ng tß╗ìa ─æß╗Ö GPS c├ích ─æiß╗âm tr╞░ß╗¢c > 100 m├⌐t chß╗ë trong v├ái gi├óy, hß╗ç thß╗æng xß║┐p v├áo trß║íng th├íi `UNCERTAIN` v├á t─âng hß╗ç sß╗æ thß║¡n trß╗ìng cß╗ºa bß╗Ö lß╗ìc.

### 5.3. Ph├ón ─æoß║ín dß╗» liß╗çu (Segmentation) & Reset State
- **Quy tß║»c ─æß╗⌐t qu├úng 120 ph├║t**: Nß║┐u khoß║úng c├ích thß╗¥i gian giß╗»a 2 bß║ún tin li├¬n tiß║┐p $\Delta t > 120\text{ ph├║t}$ (tham sß╗æ `reset_gap_minutes = 120.0`), hß╗ç thß╗æng x├íc ─æß╗ïnh xe ─æ├ú trß║úi qua thß╗¥i gian nghß╗ë d├ái kh├┤ng gi├ím s├ít.
- **H├ánh vi Reset State**:
  - X├│a trß║»ng bß╗Ö ─æß╗çm lß╗ïch sß╗¡ cß╗ºa xe.
  - Khß╗ƒi tß║ío lß║íi bß╗Ö lß╗ìc Kalman vß╗¢i gi├í trß╗ï ─æo mß╗¢i: $x_0 = \text{RawFuel}$, $P_0 = 1.0$.
  - Tr├ính hiß╗çn t╞░ß╗úng k├⌐o m╞░ß╗út mß╗Öt ─æ╞░ß╗¥ng thß║│ng giß║ú tß║ío nß╗æi giß╗»a hai thß╗¥i ─æiß╗âm c├ích nhau nhiß╗üu giß╗¥.
- **Kh├íi niß╗çm Burn-in (Chß║íy r├á)**: Khi hß╗ç thß╗æng khß╗ƒi ─æß╗Öng lß║íi hoß║╖c khi hiß╗ân thß╗ï mß╗Öt ─æoß║ín dß╗» liß╗çu lß╗ïch sß╗¡ tr├¬n Dashboard, bß╗Ö lß╗ìc cß║ºn khoß║úng **5 ΓÇô 10 ─æiß╗âm ─æß║ºu ti├¬n** ─æß╗â hiß╗çp ph╞░╞íng sai $P$ hß╗Öi tß╗Ñ vß╗ü trß║íng th├íi ß╗òn ─æß╗ïnh. Trong giai ─æoß║ín burn-in, c├íc cß╗¥ chß║Ñt l╞░ß╗úng ─æ╞░ß╗úc ─æ├ính dß║Ñu `INITIAL_LOCK`.

---

## 6. Ph├ón t├¡ch v├á gß║»n nh├ún dß╗» liß╗çu (AI Data Labeling)

### 6.1. Danh mß╗Ñc 5 nh├ún trß║íng th├íi t├¡n hiß╗çu (`SignalState`)

M├┤ h├¼nh Machine Learning (Random Forest) ─æ╞░ß╗úc huß║Ñn luyß╗çn v├á ph├ón loß║íi trß╗▒c tiß║┐p tr├¬n ─æ├║ng 5 lß╗¢p trß║íng th├íi t├¡n hiß╗çu (khß╗¢p 100% Ma trß║¡n nhß║ºm lß║½n 5x5):

| T├¬n nh├ún | ─Éß╗ïnh ngh─⌐a kß╗╣ thuß║¡t | Hiß╗çn t╞░ß╗úng vß║¡t l├╜ thß╗▒c tß║┐ |
| :--- | :--- | :--- |
| `STABLE_JITTER` | Dao ─æß╗Öng bi├¬n ─æß╗Ö nhß╗Å quanh mß╗⌐c trung b├¼nh t─⌐nh (<= 0.8 L). | Xe dß╗½ng nß╗ò m├íy, hoß║╖c cß║úm biß║┐n rung c╞í hß╗ìc khi ─æß╗ù. |
| `GRADUAL_CHANGE` | Mß╗⌐c nhi├¬n liß╗çu giß║úm tß╗½ tß╗½ v├á ─æß╗üu ─æß║╖n theo thß╗¥i gian. | Ti├¬u hao nhi├¬n liß╗çu b├¼nh th╞░ß╗¥ng khi ─æß╗Öng c╞í hoß║ít ─æß╗Öng. |
| `OSCILLATION_NOISE` | Dao ─æß╗Öng nhiß╗àu tß║ºn sß╗æ cao, ─æß╗òi h╞░ß╗¢ng li├¬n tß╗Ñc. | Xe ─æi qua ß╗ò g├á, gß╗¥ giß║úm tß╗æc, ─æ╞░ß╗¥ng gß╗ô ghß╗ü. |
| `UPWARD_SHIFT` | Mß║╖t bß║▒ng t├¡n hiß╗çu dß╗ïch chuyß╗ân t─âng ─æß╗Öt ngß╗Öt v├á duy tr├¼ mß╗⌐c mß╗¢i. | Mß╗⌐c ─æo t─âng bß╗ün vß╗»ng (b╞░ß╗¢c nhß║úy mß╗⌐c d╞░╞íng). |
| `DOWNWARD_SHIFT` | Mß║╖t bß║▒ng t├¡n hiß╗çu dß╗ïch chuyß╗ân giß║úm ─æß╗Öt ngß╗Öt v├á duy tr├¼ mß╗⌐c mß╗¢i. | Mß╗⌐c ─æo sß╗Ñt bß╗ün vß╗»ng (b╞░ß╗¢c nhß║úy mß╗⌐c ├óm). |

> **Cß║óNH B├üO QUAN TRß╗îNG Vß╗Ç RANH GIß╗ÜI NGHIß╗åP Vß╗ñ**:  
> Nh├ún `UPWARD_SHIFT` tuyß╗çt ─æß╗æi kh├┤ng ─æß╗ông ngh─⌐a vß╗¢i "Sß╗▒ kiß╗çn nß║íp dß║ºu". T╞░╞íng tß╗▒, `DOWNWARD_SHIFT` kh├┤ng ─æß╗ông ngh─⌐a vß╗¢i "Sß╗▒ kiß╗çn trß╗Öm dß║ºu". ─É├óy chß╗ë l├á nh├ún m├┤ tß║ú **h├¼nh hß╗ìc biß║┐n ─æß╗òi cß╗ºa t├¡n hiß╗çu**. Tß║ºng ß╗⌐ng dß╗Ñng ─Éß╗ü t├ái 2 sß║╜ kß║┐t hß╗úp th├¬m thß╗¥i gian dß╗½ng, vß║¡n tß╗æc trung b├¼nh v├á trß║íng th├íi bß║¡t m├íy ─æß╗â ra quyß║┐t ─æß╗ïnh kinh doanh.

### 6.2. Quy tr├¼nh g├ín nh├ún v├á tß║ío Dataset huß║Ñn luyß╗çn

```mermaid
flowchart LR
    A["Dß╗» liß╗çu Telemetry th├┤ (CSV)"] --> B["Thuß║¡t to├ín Heuristic gß╗úi ├╜ nh├ún ban ─æß║ºu"]
    B --> C["Dashboard trß╗▒c quan h├│a t├¡n hiß╗çu"]
    C --> D["Kß╗╣ s╞░ chuy├¬n gia kiß╗âm tra & ─Éiß╗üu chß╗ënh nh├ún"]
    D --> E["Dataset v├áng ─æ╞░ß╗úc duyß╗çt (Approved Ground Truth)"]
    E --> F["Huß║Ñn luyß╗çn m├┤ h├¼nh Random Forest"]
```

- **Nguy├¬n tß║»c ph├ón chia Dataset kh├┤ng r├▓ rß╗ë (No Data Leakage)**:
  - Chia tß║¡p `Train / Validation / Test` theo danh s├ích **Ph╞░╞íng tiß╗çn (VehicleID)** thay v├¼ cß║»t ngß║½u nhi├¬n theo d├▓ng thß╗¥i gian.
  - To├án bß╗Ö h├ánh tr├¼nh cß╗ºa xe kiß╗âm thß╗¡ (v├¡ dß╗Ñ: `21H-02058`, `92H-02687`) kh├┤ng bao giß╗¥ xuß║Ñt hiß╗çn trong tß║¡p huß║Ñn luyß╗çn.
- **Xß╗¡ l├╜ mß║Ñt c├ón bß║▒ng nh├ún**: Tß╗╖ lß╗ç mß║½u `STABLE_JITTER` chiß║┐m tß╗¢i > 70% tß╗òng thß╗¥i gian xe chß║íy. Hß╗ç thß╗æng ├íp dß╗Ñng trß╗ìng sß╗æ lß╗¢p (`class_weight='balanced'`) v├á giß╗¢i hß║ín mß║½u tß╗æi ─æa cho c├íc lß╗¢p ─æa sß╗æ ─æß╗â m├┤ h├¼nh nhß║íy b├⌐n vß╗¢i c├íc sß╗▒ kiß╗çn hiß║┐m gß║╖p nh╞░ `SPIKE` hay `UPWARD_SHIFT`.
- **Dß╗» liß╗çu ─æang Pending Domain Review**: C├íc ─æoß║ín t├¡n hiß╗çu bi├¬n phß╗⌐c tß║íp ─æ╞░ß╗úc ─æ├ính dß║Ñu trß║íng th├íi `pending_domain_review`. Expected curve trong golden fixture chß╗ë ─æ╞░ß╗úc cß║¡p nhß║¡t khi c├│ chß╗» k├╜ ph├¬ duyß╗çt tß╗½ chuy├¬n gia phß╗Ñ tr├ích ─æß╗ü t├ái.

---

## 7. ─Éß║╖c tr╞░ng ─æß║ºu v├áo cß╗ºa m├┤ h├¼nh (Feature Engineering)

M├┤ h├¼nh AI sß╗¡ dß╗Ñng vector 15 ─æß║╖c tr╞░ng kß╗╣ thuß║¡t, ─æ╞░ß╗úc tr├¡ch xuß║Ñt ho├án to├án theo ph╞░╞íng ph├íp **Causal Window** (chß╗ë nh├¼n vß╗ü qu├í khß╗⌐ trong phß║ím vi cß╗¡a sß╗ò $W = 5\text{ ─æiß╗âm}$ gß║ºn nhß║Ñt):

| STT | T├¬n ─æß║╖c tr╞░ng | C├┤ng thß╗⌐c t├¡nh to├ín | ─É╞ín vß╗ï | Cß╗¡a sß╗ò | Vai tr├▓ ph├ít hiß╗çn nhiß╗àu |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | `fuel_raw` | $z_t$ (Gi├í trß╗ï ─æo th├┤ hiß╗çn tß║íi) | L├¡t | 1 | Mß╗⌐c tham chiß║┐u nß╗ün tß╗⌐c thß╗¥i. |
| 2 | `fuel_pct` | $z_t / \text{CapacityEst} \times 100$ | % | 1 | Chuß║⌐n h├│a mß╗⌐c nhi├¬n liß╗çu theo dung t├¡ch xe. |
| 3 | `speed` | $v_t$ (Vß║¡n tß╗æc GPS tß╗⌐c thß╗¥i) | km/h | 1 | Nhß║¡n diß╗çn xe dß╗½ng hay ─æang chß║íy. |
| 4 | `time_gap_minutes` | $(t_t - t_{t-1}) / 60$ | Ph├║t | 2 | Ph├ít hiß╗çn ─æß╗⌐t qu├úng t├¡n hiß╗çu hoß║╖c trß╗à bß║ún tin. |
| 5 | `delta_fuel` | $z_t - z_{t-1}$ | L├¡t | 2 | Tß╗æc ─æß╗Ö biß║┐n thi├¬n tß╗⌐c thß╗¥i giß╗»a 2 chu kß╗│. |
| 6 | `delta_pct` | $\Delta z / \text{CapacityEst} \times 100$ | % | 2 | Tß╗╖ lß╗ç phß║ºn tr─âm biß║┐n ─æß╗Öng tß╗⌐c thß╗¥i. |
| 7 | `abs_delta_fuel` | $\|z_t - z_{t-1}\|$ | L├¡t | 2 | C╞░ß╗¥ng ─æß╗Ö biß║┐n ─æß╗Öng th├┤ (kh├┤ng x├⌐t chiß╗üu). |
| 8 | `rolling_std` | $\text{std}(z_{t-4}, \dots, z_t)$ | L├¡t | 5 | ─Éo mß╗⌐c ─æß╗Ö ph├ón t├ín v├á c╞░ß╗¥ng ─æß╗Ö s├│ng s├ính. |
| 9 | `local_range` | $\max(W) - \min(W)$ | L├¡t | 5 | Bi├¬n ─æß╗Ö dß║¡p dß╗ünh cß╗▒c ─æß║íi trong 5 chu kß╗│. |
| 10 | `directionality` | $\|\text{mean}(\Delta z)\| / \text{mean}(\|\Delta z\|)$ | [0, 1] | 5 | Ph├ón biß╗çt nhiß╗àu dao ─æß╗Öng ─æß╗æi xß╗⌐ng ($<0.5$) vß╗¢i xu thß║┐ thß║¡t ($>0.65$). |
| 11 | `gps_displacement` | Haversine distance $(GPS_t, GPS_{t-1})$ | M├⌐t | 2 | Dß╗ïch chuyß╗ân vß║¡t l├╜ thß╗▒c tß║┐ tr├¬n mß║╖t ─æß║Ñt. |
| 12 | `prev_median` | $\text{median}(z_{t-4}, \dots, z_{t-1})$ | L├¡t | 4 qu├í khß╗⌐ | Mß║╖t bß║▒ng tin cß║¡y tr╞░ß╗¢c thß╗¥i ─æiß╗âm x├⌐t. |
| 13 | `reversal_score` | $(z_t - z_{t-1}) \times (z_{t-1} - z_{t-2})$ | $\text{L├¡t}^2$ | 3 | Ph├ít hiß╗çn xung nhß╗ìn ─æß║úo chiß╗üu tß╗⌐c thß╗¥i (`SPIKE`). |
| 14 | `capacity_est` | Gi├í trß╗ï dung t├¡ch b├¼nh ─æang ├íp dß╗Ñng | L├¡t | 1 | Hß╗ç sß╗æ co gi├ún c├íc ng╞░ß╗íng lß╗ìc theo k├¡ch cß╗í b├¼nh. |
| 15 | `noise_sigma` | $\max(0.5, \text{CapacityEst} \times 0.002)$ | L├¡t | 1 | Ng╞░ß╗íng nhiß╗àu nß╗ün chuß║⌐n h├│a cß╗ºa ph╞░╞íng tiß╗çn. |

> **Ph├ón ─æß╗ïnh r├╡ Realtime vs Offline**: To├án bß╗Ö 15 ─æß║╖c tr╞░ng tr├¬n ─æß╗üu l├á **Causal** v├á ─æ╞░ß╗úc ph├⌐p chß║íy trong luß╗ông Realtime API. C├íc ─æß║╖c tr╞░ng phi nh├ón quß║ú (nh╞░ centered rolling mean hay forward slope) tuyß╗çt ─æß╗æi kh├┤ng ─æ╞░ß╗úc ─æ╞░a v├áo production.

---

## 8. M├┤ h├¼nh AI ph├ón loß║íi t├¡n hiß╗çu (AI Model Specifications)

- **Kiß║┐n tr├║c m├┤ h├¼nh**: **Random Forest Classifier** (Scikit-Learn).
- **Trß╗ìng sß╗æ & Cß║Ñu h├¼nh ch├¡nh thß╗⌐c**: [models/fuel_state_classifier/fuel_state_classifier.pkl](file:///D:/THUCTAP_VICOMSAT/models/fuel_state_classifier/fuel_state_classifier.pkl) c├╣ng file metadata [models/fuel_state_classifier/metadata.json](file:///D:/THUCTAP_VICOMSAT/models/fuel_state_classifier/metadata.json).
- **─Éß║ºu v├áo**: Vector 28 ─æß║╖c tr╞░ng phß║ún ├ính ─æß╗Öng hß╗ìc t├¡n hiß╗çu, thß╗æng k├¬ tr╞░ß╗út v├á ng╞░ß╗íng th├¡ch nghi dung t├¡ch xe.
- **Sß╗æ lß╗¢p ph├ón loß║íi**: **5 lß╗¢p ─æß╗Öng hß╗ìc cß╗æt l├╡i** (`UPWARD_SHIFT`, `DOWNWARD_SHIFT`, `GRADUAL_CHANGE`, `STABLE_JITTER`, `OSCILLATION_NOISE`).
- **╞»u ─æiß╗âm triß╗ân khai**:
  1. ─Éß╗Ö trß╗à suy luß║¡n (Inference Latency) cß╗▒c thß║Ñp: **< 1.5 ms/─æiß╗âm**, ho├án to├án kh├┤ng ─æ├▓i hß╗Åi GPU.
  2. Khß║ú n─âng chß╗æng Overfitting tß╗æt nhß╗¥ c╞í chß║┐ ensemble 300 c├óy quyß║┐t ─æß╗ïnh ─æß╗Öc lß║¡p.
  3. Hoß║ít ─æß╗Öng ß╗òn ─æß╗ïnh tr├¬n dß╗» liß╗çu viß╗àn th├┤ng thß╗▒c tß║┐ v├á hß╗ù trß╗ú kiß╗âm so├ít t├¡nh quan trß╗ìng cß╗ºa ─æß║╖c tr╞░ng.

### 8.1. Ma trß║¡n nhß║ºm lß║½n tß║¡p Test ─æß╗Öc lß║¡p (Held-Out 3 xe: 90H-03494, 92H-02687, Car 5)

Tß║¡p kiß╗âm thß╗¡ ─æß╗Öc lß║¡p bao gß╗ôm 23,486 mß║½u t├¡n hiß╗çu thß╗▒c tß║┐ ho├án to├án ch╞░a xuß║Ñt hiß╗çn trong qu├í tr├¼nh huß║Ñn luyß╗çn:

![Ma trß║¡n nhß║ºm lß║½n tß║¡p Test ─æß╗Öc lß║¡p](docs/images/test_held_out_confusion_matrix.svg)
*H├¼nh: Ma trß║¡n nhß║ºm lß║½n (Vector SVG) tr├¬n tß║¡p kiß╗âm thß╗¡ ─æß╗Öc lß║¡p 3 xe Unseen (Accuracy ─æß║ít 97.33%).*

| Lß╗¢p t├¡n hiß╗çu (`SignalState`) | Precision | Recall | F1-Score | Sß╗æ l╞░ß╗úng mß║½u (Support) |
| :--- | :---: | :---: | :---: | :---: |
| `UPWARD_SHIFT` (Dß╗ïch mß╗⌐c t─âng) | 0.94 | 1.00 | **0.97** | 64 |
| `DOWNWARD_SHIFT` (Dß╗ïch mß╗⌐c giß║úm) | 0.97 | 0.72 | **0.83** | 116 |
| `GRADUAL_CHANGE` (Ti├¬u hao dß╗æc) | 0.98 | 0.91 | **0.94** | 1,526 |
| `STABLE_JITTER` (ß╗ön ─æß╗ïnh / ─Éß╗ù) | 1.00 | 0.98 | **0.99** | 17,928 |
| `OSCILLATION_NOISE` (S├│ng s├ính / Nhiß╗àu) | 0.87 | 0.99 | **0.92** | 3,849 |
| **─Éß╗Ö ch├¡nh x├íc to├án bß╗Ö tß║¡p Test** | ΓÇö | ΓÇö | **Accuracy: 97.33%** | **23,483** |

---

### 8.2. Ma trß║¡n nhß║ºm lß║½n thß╗▒c tß║┐ tr├¬n xe ─æß║íi diß╗çn: Car 5 (─Éß║ºy ─æß╗º 5 trß║íng th├íi)

Xe `Car 5` (13,698 mß║½u) l├á mß║½u ph╞░╞íng tiß╗çn vß║¡n tß║úi c├│ chu tr├¼nh vß║¡n h├ánh phß╗⌐c tß║íp v├á ─æß║ºy ─æß╗º nhß║Ñt: di chuyß╗ân ─æ╞░ß╗¥ng tr╞░ß╗¥ng rung lß║»c mß║ính, c├│ c├íc ─æß╗út nß║íp nhi├¬n liß╗çu thß║¡t (`UPWARD_SHIFT`), sß╗Ñt mß╗⌐c ─æß╗Öt ngß╗Öt (`DOWNWARD_SHIFT`), ti├¬u hao dß╗æc li├¬n tß╗Ñc v├á ─æß╗ù nß╗ò m├íy:

![Ma trß║¡n nhß║ºm lß║½n xe Car 5](docs/images/cm_Car_5.svg)
*H├¼nh: Ma trß║¡n nhß║ºm lß║½n ─æß╗ïnh dß║íng Vector SVG cß╗ºa xe Car 5 (─Éß╗Ö ch├¡nh x├íc thß╗▒c tß║┐ ─æß║ít 95.58%).*

#### Chi tiß║┐t bß║úng ma trß║¡n nhß║ºm lß║½n xe Car 5:
| Thß╗▒c tß║┐ \ Dß╗▒ ─æo├ín | UPWARD_SHIFT | DOWNWARD_SHIFT | GRADUAL_CHANGE | STABLE_JITTER | OSCILLATION_NOISE | Tß╗òng mß║½u thß╗▒c tß║┐ | ─Éß╗Ö ch├¡nh x├íc |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| **UPWARD_SHIFT** | **62** | 0 | 0 | 0 | 0 | 62 | **100.00%** |
| **DOWNWARD_SHIFT** | 0 | **84** | 0 | 0 | 32 | 116 | **72.41%** |
| **GRADUAL_CHANGE** | 0 | 0 | **577** | 0 | 132 | 709 | **81.38%** |
| **STABLE_JITTER** | 0 | 0 | 0 | **8,679** | 392 | 9,071 | **95.68%** |
| **OSCILLATION_NOISE** | 4 | 3 | 30 | 13 | **3,690** | 3,740 | **98.66%** |
| **Tß╗òng dß╗▒ ─æo├ín** | 66 | 87 | 607 | 8,692 | 4,246 | **13,698** | **95.58%** |

---

### 8.3. ─É├ính gi├í to├án diß╗çn tr├¬n to├án hß║ím ─æß╗Öi (34 ph╞░╞íng tiß╗çn)

Hß╗ç thß╗æng ─æ├ú ─æ╞░ß╗úc kiß╗âm ─æß╗ïnh tr├¬n to├án bß╗Ö **34 xe** (gß╗ôm 5 xe tß╗½ `CarFuelHistory.xlsx` v├á 29 xe tß╗½ th╞░ mß╗Ñc `TienXuLy`):
- **Tß╗òng sß╗æ mß║½u kiß╗âm ─æß╗ïnh hß╗úp lß╗ç**: **410,748 ─æiß╗âm ─æo**.
- **─Éß╗Ö ch├¡nh x├íc to├án hß║ím ─æß╗Öi (Fleet Accuracy)**: **99.65%**.
- **T├ái liß╗çu ─æß╗æi so├ít chi tiß║┐t**:
  - [summary_per_vehicle.md](file:///D:/THUCTAP_VICOMSAT/reports/confusion_matrices/summary_per_vehicle.md): B├ío c├ío chi tiß║┐t bß║úng ma trß║¡n cß╗ºa to├án bß╗Ö 34 xe.
  - [reports/confusion_matrices/svg/](file:///D:/THUCTAP_VICOMSAT/reports/confusion_matrices/svg/): To├án bß╗Ö 34 file ß║únh vector SVG ri├¬ng lß║╗ tß╗½ng xe.
  - [fleet_accuracy_summary.csv](file:///D:/THUCTAP_VICOMSAT/reports/confusion_matrices/fleet_accuracy_summary.csv): Bß║úng dß╗» liß╗çu thß╗æng k├¬ ph├ón bß╗æ nh├ún v├á ─æß╗Ö ch├¡nh x├íc.

- **H├ánh vi Fallback an to├án (Safe Fallback)**:
  Nß║┐u file `.pkl` bß╗ï thiß║┐u hoß║╖c lß╗ùi m├┤i tr╞░ß╗¥ng, l├╡i `SmoothTrackingFilterEngine` sß║╜ tß╗▒ ─æß╗Öng chuyß╗ân sang c╞í chß║┐ **Heuristic Rule-based Classifier**. Bß╗Ö lß╗ìc t├¡m vß║½n tiß║┐p tß╗Ñc hoß║ít ─æß╗Öng li├¬n tß╗Ñc dß╗▒a tr├¬n c├íc ng╞░ß╗íng ─æß╗Ö lß╗çch chuß║⌐n v├á vß║¡n tß╗æc m├á kh├┤ng l├ám sß║¡p tiß║┐n tr├¼nh API.

---

## 9. Thuß║¡t to├ín Smooth-Tracking m├áu t├¡m (Core Algorithm)

─É├óy l├á th├ánh phß║ºn trung t├óm quyß║┐t ─æß╗ïnh chß║Ñt l╞░ß╗úng ─æ╞░ß╗¥ng lß╗ìc **CleanFuel**.

### 9.1. Trß║íng th├íi l╞░u trß╗» theo xe (Vehicle State Context)
Vß╗¢i mß╗ùi xe, hß╗ç thß╗æng duy tr├¼ trong bß╗Ö nhß╗¢ mß╗Öt cß║Ñu tr├║c `VehicleFilterContext`:
- Gi├í trß╗ï ╞░ß╗¢c l╞░ß╗úng Kalman $x$ v├á hiß╗çp ph╞░╞íng sai sai sß╗æ $P$.
- ─Éiß╗âm ─æo th├┤ cuß╗æi c├╣ng $z_{last}$, mß╗⌐c sß║ích cuß╗æi $x_{last}$ v├á mß╗æc thß╗¥i gian $t_{last}$.
- Bß╗Ö ─æß╗çm cß╗¡a sß╗ò tr╞░ß╗út: 5 gi├í trß╗ï nhi├¬n liß╗çu, 5 gi├í trß╗ï vß║¡n tß╗æc, 5 tß╗ìa ─æß╗Ö GPS gß║ºn nhß║Ñt.
- Trß║íng th├íi ß╗⌐ng vi├¬n b╞░ß╗¢c nhß║úy: `candidate_level`, `candidate_counter`, `candidate_direction`.
- Bß╗Ö ─æß║┐m giß╗» mß╗⌐c: `valley_hold_counter`, `zero_hold_counter`.

### 9.2. Thuß║¡t to├ín Adaptive Kalman 1 chiß╗üu
Thuß║¡t to├ín lß╗ìc Kalman 1 chiß╗üu thß╗▒c hiß╗çn qua 2 giai ─æoß║ín tß║íi mß╗ùi chu kß╗│:

1. **Giai ─æoß║ín Dß╗▒ ─æo├ín (Prediction)**:
   $$P' = P + Q$$
2. **Giai ─æoß║ín Cß║¡p nhß║¡t (Measurement Update)**:
   $$K = \frac{P'}{P' + R}$$
   $$x = x + K \times (z - x)$$
   $$P = (1 - K) \times P'$$

Trong ─æ├│:
- $z$: Gi├í trß╗ï mß╗⌐c nhi├¬n liß╗çu th├┤ ─æß║ºu v├áo (`RawFuel`).
- $x$: Gi├í trß╗ï mß╗⌐c nhi├¬n liß╗çu ─æ├ú lß╗ìc xuß║Ñt x╞░ß╗ƒng (`CleanFuel`).
- $K$: Hß╗ç sß╗æ khuß║┐ch ─æß║íi Kalman (Kalman Gain, $0 \le K \le 1$).
- **$Q$ (Process Noise Covariance)**: Mß╗⌐c ─æß╗Ö tin t╞░ß╗ƒng v├áo sß╗▒ thay ─æß╗òi vß║¡t l├╜ thß╗▒c tß║┐ cß╗ºa hß╗ç thß╗æng.
- **$R$ (Measurement Noise Covariance)**: Mß╗⌐c ─æß╗Ö nghi ngß╗¥ sai sß╗æ ─æo cß╗ºa cß║úm biß║┐n.

### 9.3. Bß║úng cß║Ñu h├¼nh th├¡ch ß╗⌐ng Q v├á R (Tr├¡ch xuß║Ñt chuß║⌐n tß╗½ `config.py`)

Hß╗ç thß╗æng ─æiß╗üu chß╗ënh ─æß╗Öng $Q$ v├á $R$ theo tß╗½ng trß║íng th├íi cß╗Ñ thß╗â ─æß╗â ─æß║ít ─æ╞░ß╗úc sß╗▒ c├ón bß║▒ng tß╗æi ╞░u:

| Trß║íng th├íi vß║¡n h├ánh | Mß╗Ñc ti├¬u xß╗¡ l├╜ | Gi├í trß╗ï $R$ | Gi├í trß╗ï $Q$ | Kalman Gain ($K$) | ├¥ ngh─⌐a ß╗⌐ng xß╗¡ |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Nhiß╗àu rung khi ─æß╗ù (`parked`)** | Giß╗» t─⌐nh tuyß╗çt ─æß╗æi | **35.0** | **0.03** | Rß║Ñt nhß╗Å (~0.02) | Kh├│a chß║╖t ─æ╞░ß╗¥ng t├¡m, lß╗ìc sß║ích dao ─æß╗Öng nhß╗Å. |
| **Xe di chuyß╗ân b├¼nh th╞░ß╗¥ng (`moving`)** | B├ím ti├¬u hao m╞░ß╗út m├á | **45.0** | **0.08** | Vß╗½a phß║úi (~0.05) | Triß╗çt ti├¬u dß║¡p dß╗ünh khi chß║íy xe. |
| **Vß║¡n ─æß╗Öng thß║Ñp (`low_motion`)** | ß╗ön ─æß╗ïnh mß║╖t bß║▒ng | **28.0** | **0.06** | Trung b├¼nh (~0.06) | Th├¡ch ß╗⌐ng khi xe di chuyß╗ân chß║¡m trong b├úi. |
| **Nhiß╗àu cß╗▒c mß║ính (`strong_noise`)** | Khß╗¡ xung s├│ng s├ính | **250.0** | **0.01** | Cß╗▒c nhß╗Å (~0.005) | ─É╞░ß╗¥ng t├¡m gß║ºn nh╞░ nß║▒m ngang, phß╗¢t lß╗¥ dao ─æß╗Öng. |
| **Dao ─æß╗Öng nhß║╣ (`mild_noise`)** | L├ám phß║│ng nhß║╣ nh├áng | **18.0** | **0.12** | Linh hoß║ít (~0.15) | B├ím s├ít dß╗» liß╗çu sß║ích. |
| **Xu h╞░ß╗¢ng giß║úm r├╡ (`directional`)** | B├ím s├ít sß╗Ñt giß║úm | **8.0** | **1.00** | Lß╗¢n (~0.40) | Giß║úm ─æß╗Ö trß╗à khi xe ti├¬u hao thß║¡t. |
| **Xu h╞░ß╗¢ng bß╗ün vß╗»ng (`robust_trend`)**| B├ím tß╗⌐c thß╗¥i | **6.0** | **1.50** | Rß║Ñt lß╗¢n (~0.60) | B├ím dß╗æc ti├¬u hao mß║ính m├á kh├┤ng bß╗ï trß╗à. |
| **GPS m├óu thuß║½n (`gps_conflict`)** | Thß║¡n trß╗ìng bß║úo vß╗ç | $\times 1.30$ | $\times 0.75$ | Giß║úm | Tß╗▒ ─æß╗Öng t─âng $R$, giß║úm $Q$ khi vß║¡n tß╗æc v├á GPS lß╗çch pha. |

![H├ánh vi th├¡ch ß╗⌐ng ─æß╗Öng cß╗ºa bß╗Ö lß╗ìc Adaptive Kalman](docs/images/adaptive_kalman_behavior.png)
*H├¼nh: C╞í chß║┐ ─æiß╗üu tiß║┐t ─æß╗Öng hß╗ç sß╗æ lß╗ìc th├¡ch ß╗⌐ng (Q, R) b├ím s├ít c├íc dß║íng vß║¡n ─æß╗Öng thß╗▒c tß║┐ cß╗ºa ph╞░╞íng tiß╗çn.*

### 9.4. S╞í ─æß╗ô c├óy quyß║┐t ─æß╗ïnh ph├ón nh├ính xß╗¡ l├╜

```mermaid
flowchart TD
    A["Nhß║¡n ─æiß╗âm ─æo mß╗¢i (RawFuel, Time, Speed, GPS)"] --> B{"Raw c├│ hß╗úp lß╗ç kh├┤ng?"}
    B -- "Kh├┤ng (<=0 hoß║╖c NaN)" --> C["Giß╗» nguy├¬n CleanFuel c┼⌐ (DROPOUT_ZERO_HOLD)"]
    B -- "C├│" --> D{"C├│ phß║úi ─æß╗Öt biß║┐n Spike ─æ╞ín lß║╗?"}
    D -- "─É├║ng (Nhß║úy vß╗ìt rß╗ôi hß╗ôi vß╗ü ngay)" --> E["Triß╗çt ti├¬u xung, giß╗» Clean c┼⌐ (SPIKE_SUPPRESS)"]
    D -- "Sai" --> F{"─Éß╗òi mß╗⌐c lß╗¢n (|delta| >= LevelShiftFloor)?"}
    F -- "T─âng ─æß╗Öt ngß╗Öt" --> G["Chß╗¥ x├íc nhß║¡n 4 ─æiß╗âm li├¬n tiß║┐p (UPWARD_HOLD)"]
    F -- "Giß║úm ─æß╗Öt ngß╗Öt" --> H["Chß╗¥ x├íc nhß║¡n 3 ─æiß╗âm li├¬n tiß║┐p (DOWNWARD_CONFIRM)"]
    F -- "Kh├┤ng ─æß╗òi mß╗⌐c lß╗¢n" --> I{"C├│ xu h╞░ß╗¢ng giß║úm r├╡ rß╗çt?"}
    I -- "─É├║ng (Directionality >= 0.65)" --> J["├üp dß╗Ñng Q lß╗¢n (1.0), R thß║Ñp (8.0) ─æß╗â b├ím dß╗æc"]
    I -- "Sai (Dao ─æß╗Öng ─æß╗æi xß╗⌐ng)" --> K["├üp dß╗Ñng Q nhß╗Å (0.01), R lß╗¢n (250.0) ─æß╗â l├ám phß║│ng"]
    G --> L["Cß║¡p nhß║¡t Kalman & Trß║ú vß╗ü CleanFuel"]
    H --> L
    J --> L
    K --> L
    E --> L
    C --> L
```

### 9.5. Chi tiß║┐t c├íc nh├ính xß╗¡ l├╜ ─æß║╖c th├╣
1. **Khß╗¡ sß╗Ñt chß╗» U (Valley Recovery)**: Khi cß║úm biß║┐n sß╗Ñt s├óu tß║ím thß╗¥i rß╗ôi hß╗ôi phß╗Ñc lß║íi nß╗ün c┼⌐ trong 1ΓÇô3 ─æiß╗âm (do phanh hoß║╖c dß╗æc), thuß║¡t to├ín giß╗» nguy├¬n ─æ╞░ß╗¥ng t├¡m ß╗ƒ mß║╖t bß║▒ng tr╞░ß╗¢c ─æ├│, ho├án to├án triß╗çt ti├¬u ─æ├íy v├╡ng chß╗» U.
2. **Khß╗¡ ─æß╗ôi t─âng ngß║»n (Hill Anomaly)**: Mß╗⌐c ─æo vß╗ìt l├¬n rß╗ôi tß╗Ñt vß╗ü trong 1ΓÇô3 ─æiß╗âm bß╗ï c├┤ lß║¡p, kh├┤ng cho ph├⌐p k├⌐o ─æ╞░ß╗¥ng t├¡m t─âng giß║ú.
3. **X├íc nhß║¡n mß║╖t bß║▒ng t─âng bß╗ün vß╗»ng (`UPWARD_HOLD`)**: Mß╗⌐c ─æo t─âng cao phß║úi duy tr├¼ li├¬n tß╗Ñc qua `upward_hold_steps = 4` ─æiß╗âm mß╗¢i ─æ╞░ß╗úc c├┤ng nhß║¡n l├á mß║╖t bß║▒ng mß╗¢i. Nß║┐u trong 4 ─æiß╗âm n├áy raw quay vß╗ü nß╗ün c┼⌐, candidate bß╗ï hß╗ºy bß╗Å.
4. **X├íc nhß║¡n mß║╖t bß║▒ng giß║úm bß╗ün vß╗»ng (`DOWNWARD_CONFIRM`)**: Mß╗⌐c ─æo sß╗Ñt s├óu phß║úi duy tr├¼ ß╗òn ─æß╗ïnh qua `downward_confirm_points = 3` ─æiß╗âm mß╗¢i ─æ╞░ß╗úc k├⌐o dß╗æc ─æ╞░ß╗¥ng t├¡m xuß╗æng mß║╖t bß║▒ng mß╗¢i.

---

## 10. T├¡nh Real-time, Causal v├á ─Éß╗Ö trß╗à ─æ├ính ─æß╗òi

### 10.1. Nguy├¬n l├╜ Bß║Ñt biß║┐n Causal (No-Lookahead Invariant)
- Hß╗ç thß╗æng xß╗¡ l├╜ theo m├┤ h├¼nh **Online Streaming**: Mß╗ùi khi mß╗Öt bß║ún tin ─æß║┐n, kß║┐t quß║ú `CleanFuel` ─æ╞░ß╗úc t├¡nh to├ín v├á trß║ú vß╗ü ngay lß║¡p tß╗⌐c.
- **Kh├┤ng bao giß╗¥ sß╗¡a ─æß╗òi qu├í khß╗⌐**: ─Éß║ºu ra ─æ├ú ph├ít h├ánh cho c├íc hß╗ç thß╗æng ph├¡a sau l├á bß║Ñt biß║┐n, kh├┤ng c├│ c╞í chß║┐ "sß╗¡a lß║íi ─æiß╗âm tr╞░ß╗¢c khi nhß║¡n th├¬m ─æiß╗âm sau".

### 10.2. Bß║ún chß║Ñt sß╗▒ ─æ├ính ─æß╗òi (Latency Trade-off)
─Éß╗â ph├ón biß╗çt ─æ╞░ß╗úc giß╗»a **nhiß╗àu s├│ng s├ính tß║ím thß╗¥i** v├á **mß╗Öt b╞░ß╗¢c nhß║úy mß╗⌐c thß╗▒c sß╗▒**, hß╗ç thß╗æng bß║»t buß╗Öc phß║úi chß╗¥ tß╗½ 2 ─æß║┐n 4 ─æiß╗âm ─æo ─æß╗â t├¡ch l┼⌐y bß║▒ng chß╗⌐ng:
- Vß╗¢i chu kß╗│ gß╗¡i mß║½u chuß║⌐n 2 ph├║t/─æiß╗âm, ─æß╗Ö trß╗à x├íc nhß║¡n mß╗Öt b╞░ß╗¢c nhß║úy mß║╖t bß║▒ng l├á **4 ΓÇô 8 ph├║t**.
- Trong khoß║úng thß╗¥i gian chß╗¥ x├íc nhß║¡n n├áy, hß╗ç thß╗æng chß╗º ─æß╗Öng giß╗» `CleanFuel` ß╗ƒ mß╗⌐c an to├án tr╞░ß╗¢c ─æ├│.
- ─É├óy l├á **sß╗▒ ─æ├ính ─æß╗òi vß║¡t l├╜ bß║»t buß╗Öc**: Nß║┐u muß╗æn lß╗ìc sß║ích 100% c├íc ─æß╗ënh s├│ng s├ính giß║ú, kh├┤ng mß╗Öt hß╗ç thß╗æng causal n├áo c├│ thß╗â b├ím ngay lß║¡p tß╗⌐c tß║íi ─æiß╗âm ─æß║ºu ti├¬n.

### 10.3. Bß║úng minh hß╗ìa chuß╗ùi phß║ún ß╗⌐ng 20 ─æiß╗âm thß╗▒c tß║┐

```text
Chuß╗ùi minh hß╗ìa: Sß╗Ñt chß╗» U giß║ú (─Éiß╗âm 4-6) sau ─æ├│ Xe ti├¬u hao ─æß╗üu (─Éiß╗âm 9-16):
─Éiß╗âm | RawFuel (L) | CleanFuel (L) | QualityFlag         | Diß╗àn giß║úi h├ánh vi
  1  |   200.0     |    200.0      | INITIAL_LOCK        | Khß╗ƒi tß║ío gi├í trß╗ï ban ─æß║ºu
  2  |   200.2     |    200.0      | KALMAN_SMOOTH       | Lß╗ìc rung nhß║╣ khi ─æß╗ù
  3  |   199.8     |    200.0      | KALMAN_SMOOTH       | Lß╗ìc rung nhß║╣ khi ─æß╗ù
  4  |   182.0     |    200.0      | VALLEY_HOLD         | Sß╗Ñt giß║ú do phanh gß║Ñp (Bß║»t ─æß║ºu chß╗» U)
  5  |   181.5     |    200.0      | VALLEY_HOLD         | ─É├íy chß╗» U, giß╗» nguy├¬n mß╗⌐c sß║ích
  6  |   183.0     |    200.0      | VALLEY_HOLD         | ─Éang hß╗ôi phß╗Ñc
  7  |   199.5     |    199.8      | RECOVERY_SMOOTH     | Raw ─æ├ú vß╗ü nß╗ün c┼⌐, triß╗çt ti├¬u ho├án to├án chß╗» U
  8  |   199.6     |    199.7      | KALMAN_SMOOTH       | Trß║íng th├íi ß╗òn ─æß╗ïnh b├¼nh th╞░ß╗¥ng
  9  |   198.5     |    199.2      | TREND_TRACKING      | Xe bß║»t ─æß║ºu chß║íy, ph├ít hiß╗çn xu h╞░ß╗¢ng giß║úm
 10  |   197.6     |    198.3      | TREND_TRACKING      | B├ím s├ít dß╗æc ti├¬u hao thß║¡t
 11  |   196.8     |    197.4      | TREND_TRACKING      | B├ím s├ít dß╗æc ti├¬u hao thß║¡t
 12  |   196.0     |    196.6      | TREND_TRACKING      | B├ím s├ít dß╗æc ti├¬u hao thß║¡t
 13  |   195.1     |    195.7      | TREND_TRACKING      | B├ím s├ít dß╗æc ti├¬u hao thß║¡t
 14  |   194.2     |    194.8      | TREND_TRACKING      | B├ím s├ít dß╗æc ti├¬u hao thß║¡t
 15  |   193.5     |    194.0      | TREND_TRACKING      | B├ím s├ít dß╗æc ti├¬u hao thß║¡t
 16  |   192.8     |    193.3      | TREND_TRACKING      | Ti├¬u hao ß╗òn ─æß╗ïnh
 17  |     0.0     |    193.3      | DROPOUT_ZERO_HOLD   | Cß║úm biß║┐n rß╗¢t vß╗ü 0L do gi├ín ─æoß║ín d├óy
 18  |     0.0     |    193.3      | DROPOUT_ZERO_HOLD   | Tiß║┐p tß╗Ñc kh├│a mß╗⌐c sß║ích an to├án
 19  |   192.2     |    192.6      | RECOVERY_SMOOTH     | Cß║úm biß║┐n c├│ lß║íi, h├▓a nhß╗ïp m╞░ß╗út m├á
 20  |   191.9     |    192.2      | TREND_TRACKING      | Trß╗ƒ lß║íi luß╗ông b├ím b├¼nh th╞░ß╗¥ng
```

---

## 11. ─É├ính gi├í trß║íng th├íi vß║¡n ─æß╗Öng (Speed & GPS Motion)

Nhß║▒m loß║íi bß╗Å hiß╗çn t╞░ß╗úng sai sß╗æ do GPS ─æß╗⌐ng y├¬n nhß║úy ─æiß╗âm, hß╗ç thß╗æng kß║┐t hß╗úp th├┤ng tin ─æa k├¬nh giß╗»a ─æß╗ông hß╗ô vß║¡n tß╗æc v├á dß╗ïch chuyß╗ân tß╗ìa ─æß╗Ö ─æß╗â xuß║Ñt ra 3 trß║íng th├íi `MotionState`:

| Trß║íng th├íi `MotionState` | ─Éiß╗üu kiß╗çn k├¡ch hoß║ít kß╗╣ thuß║¡t | ß╗¿ng xß╗¡ cß╗ºa bß╗Ö lß╗ìc |
| :--- | :--- | :--- |
| `MOVING` | $\text{Speed} \ge 5.0\text{ km/h}$ HOß║╢C dß╗ïch chuyß╗ân GPS cß╗¡a sß╗ò $\ge 25.0\text{ m}$. | Bß╗Ö lß╗ìc hiß╗âu xe ─æang chß║íy thß║¡t; cho ph├⌐p b├ím dß╗æc ti├¬u hao nhi├¬n liß╗çu. |
| `LOW_MOTION` | $\text{Speed} \le 1.0\text{ km/h}$ V├Ç to├án bß╗Ö GPS 3 ─æiß╗âm gß║ºn nhß║Ñt nß║▒m trong b├ín k├¡nh $30.0\text{ m}$. | Bß╗Ö lß╗ìc kh├│a chß║╖t mß║╖t bß║▒ng t─⌐nh, t─âng $R$ ─æß╗â dß║¡p tß║»t dao ─æß╗Öng phao khi ─æß╗ù nß╗ò m├íy. |
| `UNCERTAIN` | Vß║¡n tß╗æc v├á GPS m├óu thuß║½n (VD: Speed = 0 nh╞░ng tß╗ìa ─æß╗Ö nhß║úy > 50m). | Tß╗▒ ─æß╗Öng nh├ón hß╗ç sß╗æ thß║¡n trß╗ìng: t─âng $R$ th├¬m 30%, giß║úm $Q$ ─æi 25%. |

> **Quy ╞░ß╗¢c kß╗╣ thuß║¡t**: Hß╗ç thß╗æng ─æß╗ïnh danh trß║íng th├íi l├á `LOW_MOTION`, tuyß╗çt ─æß╗æi kh├┤ng khß║│ng ─æß╗ïnh xe ─æang "─Éß╗ù" hay "Tß║»t m├íy" v├¼ thiß║┐u k├¬nh t├¡n hiß╗çu ch├ón kh├│a ─æiß╗çn ACC/RPM.

---

## 12. ─Éß║╖c tß║ú giao diß╗çn ─æß║ºu ra (Output Contract)

Mß╗ùi lß║ºn gß╗ìi API xß╗¡ l├╜ ─æiß╗âm, hß╗ç thß╗æng trß║ú vß╗ü cß║Ñu tr├║c JSON ─æß╗ông nhß║Ñt:

### 12.1. Bß║úng ─æß║╖c tß║ú c├íc tr╞░ß╗¥ng dß╗» liß╗çu ─æß║ºu ra

| T├¬n tr╞░ß╗¥ng | Kiß╗âu dß╗» liß╗çu | ├¥ ngh─⌐a kß╗╣ thuß║¡t |
| :--- | :--- | :--- |
| `VehicleID` | `string` | ─Éß╗ïnh danh xe ─æ╞░ß╗úc xß╗¡ l├╜. |
| `FuelTime` | `string` | Mß╗æc thß╗¥i gian cß╗ºa ─æiß╗âm ─æo hiß╗çn tß║íi. |
| `RawFuel` | `float` | Gi├í trß╗ï ─æo th├┤ ban ─æß║ºu (L├¡t). |
| `CleanFuel` | `float` | **Gi├í trß╗ï nhi├¬n liß╗çu ─æ├ú lß╗ìc sß║ích (L├¡t) - D├╣ng hiß╗ân thß╗ï cho kh├ích h├áng.** |
| `SignalState` | `string` | Nh├ún h├¼nh hß╗ìc t├¡n hiß╗çu tß╗½ AI (`STABLE_JITTER`, `SLOSHING`, `UPWARD_SHIFT`...). |
| `QualityFlag` | `string` | H├ánh vi bß╗Ö lß╗ìc ─æ├ú thß╗▒c thi (`KALMAN_SMOOTH`, `VALLEY_HOLD`, `SPIKE_SUPPRESS`...). |
| `MotionState` | `string` | Trß║íng th├íi chuyß╗ân ─æß╗Öng (`MOVING`, `LOW_MOTION`, `UNCERTAIN`). |
| `MotionConfidence` | `float` | ─Éß╗Ö tin cß║¡y cß╗ºa ─æ├ính gi├í chuyß╗ân ─æß╗Öng ($0.0 \to 1.0$). |
| `GpsDisplacementMeters` | `float` | Dß╗ïch chuyß╗ân tß╗ìa ─æß╗Ö so vß╗¢i ─æiß╗âm tr╞░ß╗¢c (m├⌐t). |
| `LatencyMs` | `float` | Thß╗¥i gian t├¡nh to├ín xß╗¡ l├╜ ─æiß╗âm tß║íi server (mili-gi├óy). |

### 12.2. V├¡ dß╗Ñ Request & Response chuß║⌐n

**Request (`POST /api/v1/fuel/clean-point`)**:
```json
{
  "VehicleID": "21H-02058",
  "FuelTime": "2026-08-13T10:54:00",
  "FuelLevel": 175.2,
  "Speed": 32.0,
  "Lat": 21.0285,
  "Lng": 105.8542,
  "CapacityEst": 200.0
}
```

**Response (HTTP 200 OK)**:
```json
{
  "VehicleID": "21H-02058",
  "FuelTime": "2026-08-13T10:54:00",
  "RawFuel": 175.2,
  "CleanFuel": 176.05,
  "SignalState": "STABLE_JITTER",
  "QualityFlag": "KALMAN_SMOOTH",
  "MotionState": "MOVING",
  "MotionConfidence": 0.95,
  "GpsDisplacementMeters": 45.2,
  "LatencyMs": 1.24
}
```

---

## 13. H╞░ß╗¢ng dß║½n t├¡ch hß╗úp API v├á SDK

### 13.1. Danh mß╗Ñc Endpoints ch├¡nh thß╗⌐c

| Method | Endpoint | Chß╗⌐c n─âng |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Kiß╗âm tra sß╗⌐c khß╗Åe service, trß║íng th├íi bß╗Ö nhß╗¢ v├á kß║┐t nß╗æi Redis. |
| `POST` | `/api/v1/fuel/clean-point` | Tiß║┐p nhß║¡n v├á l├ám sß║ích 1 ─æiß╗âm dß╗» liß╗çu thß╗¥i gian thß╗▒c. |
| `POST` | `/api/v1/fuel/clean-batch` | Tiß║┐p nhß║¡n danh s├ích nhiß╗üu ─æiß╗âm (xß╗¡ l├╜ tuß║ºn tß╗▒ nß╗Öi bß╗Ö theo xe). |
| `POST` | `/api/v1/vehicles/{vehicle_id}/reset-state` | X├│a trß║»ng state cß╗ºa mß╗Öt xe (khi thay b├¼nh hoß║╖c ─æß╗òi thiß║┐t bß╗ï). |
| `GET` | `/api/v1/vehicles/active-contexts` | Liß╗çt k├¬ danh s├ích c├íc xe ─æang l╞░u context trong RAM/Redis. |

### 13.2. M├ú mß║½u t├¡ch hß╗úp ─æa ng├┤n ngß╗»

#### V├¡ dß╗Ñ gß╗ìi bß║▒ng cURL (Linux / Windows PowerShell):
```bash
curl -X POST "http://localhost:8000/api/v1/fuel/clean-point" \
     -H "Content-Type: application/json" \
     -d '{
       "VehicleID": "29E-45520",
       "FuelTime": "2026-08-27T10:00:00",
       "FuelLevel": 105.2,
       "Speed": 45.0,
       "Lat": 21.0285,
       "Lng": 105.8542,
       "CapacityEst": 200.0
     }'
```

#### V├¡ dß╗Ñ nh├║ng Python SDK trß╗▒c tiß║┐p (Kh├┤ng qua mß║íng HTTP):
```python
from src.sdk.fuel_cleaner import FuelCleanerEngine

# Khß╗ƒi tß║ío engine 1 lß║ºn duy nhß║Ñt trong ß╗⌐ng dß╗Ñng
cleaner = FuelCleanerEngine()

result = cleaner.clean_point(
    vehicle_id="29E-45520",
    timestamp="2026-08-27 10:00:00",
    raw_fuel=105.2,
    speed=45.0,
    lat=21.0285,
    lng=105.8542,
    capacity_est=200.0
)

print(f"Mß╗⌐c nhi├¬n liß╗çu sß║ích: {result['clean_fuel']:.2f} L | Trß║íng th├íi: {result['signal_state']}")
```

#### V├¡ dß╗Ñ gß╗ìi tß╗½ C# (.NET 8):
```csharp
using System.Net.Http.Json;

var client = new HttpClient { BaseAddress = new Uri("http://localhost:8000") };
var payload = new {
    VehicleID = "29E-45520",
    FuelTime = DateTime.UtcNow.ToString("o"),
    FuelLevel = 105.2,
    Speed = 45.0,
    CapacityEst = 200.0
};

var response = await client.PostAsJsonAsync("/api/v1/fuel/clean-point", payload);
if (response.IsSuccessStatusCode) {
    var data = await response.Content.ReadFromJsonAsync<CleanFuelResult>();
    Console.WriteLine($"CleanFuel: {data.CleanFuel} L");
}
```

---

## 14. H├áng ─æß╗úi ─æß╗ông thß╗¥i v├á Quß║ún trß╗ï State (Concurrency & State Management)

### 14.1. Kiß║┐n tr├║c Kh├│a an to├án theo xe (Per-Vehicle Thread Safety)
- ─Éß╗â ─æß║úm bß║úo t├¡nh chß║Ñt Causal, c├íc bß║ún tin cß╗ºa **c├╣ng mß╗Öt ph╞░╞íng tiß╗çn** phß║úi ─æ╞░ß╗úc xß╗¡ l├╜ tuß║ºn tß╗▒ nghi├¬m ngß║╖t (FIFO).
- `VehicleQueueManager` sß╗¡ dß╗Ñng c╞í chß║┐ kh├│a ph├ón t├ích:
  - **Xe A v├á Xe B** ─æ╞░ß╗úc xß╗¡ l├╜ song song tr├¬n c├íc luß╗ông CPU kh├íc nhau (Full Concurrency).
  - **Hai ─æiß╗âm cß╗ºa c├╣ng Xe A** sß║╜ tß╗▒ ─æß╗Öng xß║┐p h├áng v├á xß╗¡ l├╜ lß║ºn l╞░ß╗út, loß║íi bß╗Å 100% rß╗ºi ro Race Condition l├ám sai lß╗çch trß║íng th├íi Kalman.

```mermaid
flowchart TD
    In["C├íc bß║ún tin Telemetry gß╗¡i ─æß║┐n API"] --> QM["VehicleQueueManager (Ph├ón luß╗ông theo VehicleID)"]
    QM --> QA["Queue ri├¬ng Xe A (Kh├│a Lock A)"]
    QM --> QB["Queue ri├¬ng Xe B (Kh├│a Lock B)"]
    QM --> QC["Queue ri├¬ng Xe C (Kh├│a Lock C)"]
    QA --> EA["Xß╗¡ l├╜ tuß║ºn tß╗▒ ─æiß╗âm t1 -> t2 -> t3"]
    QB --> EB["Xß╗¡ l├╜ song song ─æß╗Öc lß║¡p"]
    QC --> EC["Xß╗¡ l├╜ song song ─æß╗Öc lß║¡p"]
```

### 14.2. Quß║ún l├╜ l╞░u trß╗» State (RAM vs Redis)
- **M├┤i tr╞░ß╗¥ng Demo / Single-instance**: Sß╗¡ dß╗Ñng `STATE_BACKEND=memory`. Context l╞░u trong RAM, truy xuß║Ñt cß╗▒c nhanh (< 0.1 ms).
- **M├┤i tr╞░ß╗¥ng Sß║ún xuß║Ñt / Multi-instance**: Sß╗¡ dß╗Ñng `STATE_BACKEND=redis`. To├án bß╗Ö context ─æ╞░ß╗úc serialize dß║íng JSON l╞░u tr├¬n Redis vß╗¢i thß╗¥i gian sß╗æng `STATE_TTL_SECONDS=259200` (3 ng├áy kh├┤ng c├│ bß║ún tin sß║╜ tß╗▒ giß║úi ph├│ng).
- **Nguy├¬n tß║»c mß╗ƒ rß╗Öng ngang (Horizontal Scaling)**: Khi triß╗ân khai nhiß╗üu container API sau Load Balancer, bß║»t buß╗Öc phß║úi cß║Ñu h├¼nh **Hash-based Routing (Sticky Session)** theo `VehicleID` ß╗ƒ tß║ºng Gateway (Nginx / HAProxy / Traefik) ─æß╗â ─æß║úm bß║úo to├án bß╗Ö bß║ún tin cß╗ºa mß╗Öt xe ─æi v├áo c├╣ng mß╗Öt worker queue.

---

## 15. Kiß╗âm thß╗¡ Golden Segments v├á ─É├ính gi├í KPI

### 15.1. Triß║┐t l├╜ Golden Test tß╗½ dß╗» liß╗çu thß╗▒c tß║┐
Hß╗ç thß╗æng kh├┤ng kiß╗âm thß╗¡ tr├¬n dß╗» liß╗çu ngß║½u nhi├¬n giß║ú lß║¡p m├á sß╗¡ dß╗Ñng **8 ─æoß║ín dß╗» liß╗çu v├áng tr├¡ch xuß║Ñt trß╗▒c tiß║┐p tß╗½ c├íc xe chß║íy thß╗▒c tß║┐** cß╗ºa doanh nghiß╗çp (`tests/fixtures/golden_fuel_segments.json`).
- Mß╗ùi ─æoß║ín fixture ─æß║íi diß╗çn cho mß╗Öt ca bi├¬n ─æiß╗ân h├¼nh: xe dß╗½ng nß╗ò m├íy, xe ─æß╗ò ─æ├¿o, sß╗Ñt chß╗» U khi phanh, chß║íy ti├¬u hao ─æß╗üu tr├¬n cao tß╗æc, ─æß╗⌐t qu├úng mß║Ñt t├¡n hiß╗çu vß╗ü 0L.
- **Quy tß║»c bß║Ñt biß║┐n**: Tuyß╗çt ─æß╗æi kh├┤ng bao giß╗¥ ─æ╞░ß╗úc sß╗¡a ─æ╞░ß╗¥ng kß║┐t quß║ú kß╗│ vß╗ìng (`expected_clean`) trong golden test chß╗ë ─æß╗â l├ám test pass. Mß╗ìi sß╗▒ thay ─æß╗òi ─æ╞░ß╗¥ng chuß║⌐n ─æß╗üu ─æ├▓i hß╗Åi ─æ├ính gi├í lß║íi thuß║¡t to├ín.

### 15.2. B├ío c├ío kß║┐t quß║ú kiß╗âm thß╗¡ v├á KPI thß╗▒c tß║┐

D╞░ß╗¢i ─æ├óy l├á kß║┐t quß║ú kiß╗âm thß╗¡ thß╗▒c tß║┐ tß╗½ bß╗Ö test tß╗▒ ─æß╗Öng cß╗ºa hß╗ç thß╗æng:

```text
======================= Tß╗öNG Hß╗óP KIß╗éM THß╗¼ Hß╗å THß╗ÉNG =======================
- Tß╗òng sß╗æ Unit & Regression Tests: 83 / 83 tests PASSED (100%)
- Tß╗òng sß╗æ Golden Segments kiß╗âm thß╗¡: 8 ─æoß║ín thß╗▒c tß║┐ (89 ─æiß╗âm ─æo)
- Sß╗æ l╞░ß╗úng kiß╗âm tra h├ánh vi (Behavior Checks): 18 / 19 checks PASSED
- Thß╗¥i gian phß║ún hß╗ôi xß╗¡ l├╜ (Latency P50): 33.59 ms
- Thß╗¥i gian phß║ún hß╗ôi xß╗¡ l├╜ (Latency P95): 37.83 ms
- N─âng lß╗▒c xß╗¡ l├╜ (Throughput): ~31.8 ─æiß╗âm/gi├óy tr├¬n 1 worker
==========================================================================
```

#### Chi tiß║┐t KPI tß╗½ng ph├ón ─æoß║ín thß╗▒c tß║┐:

| ─Éoß║ín kiß╗âm thß╗¡ (Segment) | Hiß╗çn t╞░ß╗úng thß╗▒c tß║┐ | Giß║úm nhiß╗àu (Noise Red.) | ─Éß╗Ö lß╗çch chuß║⌐n (MAE) | Kß║┐t quß║ú kiß╗âm tra |
| :--- | :--- | :---: | :---: | :---: |
| `21H-02058_stationary_noise_pulse` | Xe dß╗½ng, nhiß╗àu dao ─æß╗Öng mß║ính | **95.99%** | 0.0 L | 1 / 2 |
| `21H-03221_short_valley_recovery` | Sß╗Ñt chß╗» U giß║ú do dß╗æc rß╗ôi hß╗ôi phß╗Ñc | **83.87%** | 0.0 L | 3 / 3 (─Éß║ít) |
| `92H-02687_steady_moving_consumption`| Xe chß║íy ─æ╞░ß╗¥ng d├ái, ti├¬u hao ─æß╗üu | Trend chuß║⌐n | 0.0 L | 3 / 3 (─Éß║ít) |
| `29E-44284_noisy_moving_consumption` | Xe chß║íy ─æ╞░ß╗¥ng gß╗ô ghß╗ü, s├│ng s├ính lß╗¢n | **39.99%** | 0.0 L | 3 / 3 (─Éß║ít) |
| `15H-08128_zero_dropout_hold` | Cß║úm biß║┐n r╞íi vß╗ü 0 L ─æß╗Öt ngß╗Öt | **Kh├│a sß║ích 100%**| 0.0 L | 2 / 2 (─Éß║ít) |
| `21H-03221_stationary_gps_cluster` | Dß╗½ng nß╗ò m├íy, cß╗Ñm GPS ─æß╗⌐ng y├¬n | Khß╗¡ rung | 0.0 L | 2 / 2 (─Éß║ít) |
| `21H-03221_u_shape_with_gps` | Phanh gß║Ñp kß║┐t hß╗úp GPS dß╗ïch chuyß╗ân | **82.52%** | 0.0 L | 2 / 2 (─Éß║ít) |
| `21H-03221_stationary_gps_jump` | Xe dß╗½ng nh╞░ng GPS nhß║úy bß║Ñt th╞░ß╗¥ng | **98.99%** | 0.0 L | 2 / 2 (─Éß║ít) |

> **B├ío c├ío trung thß╗▒c vß╗ü ca kiß╗âm tra ch╞░a ─æß║ít (`18/19`)**:  
> Tß║íi case `21H-02058_stationary_noise_pulse`, chß╗ë sß╗æ `max_clean_span` (bi├¬n ─æß╗Ö dao ─æß╗Öng lß╗¢n nhß║Ñt cß╗ºa ─æ╞░ß╗¥ng sß║ích) ─æß║ít mß╗⌐c $0.92\text{ L}$, h╞íi v╞░ß╗út nhß║╣ so vß╗¢i ng╞░ß╗íng kß╗│ vß╗ìng khß║»t khe l├á $0.80\text{ L}$ (do xe gß║╖p xung nhiß╗àu bi├¬n ─æß╗Ö tß╗¢i $25\text{ L}$). Tuy nhi├¬n, tß╗╖ lß╗ç giß║úm nhiß╗àu chung vß║½n ─æß║ít tß╗¢i **95.99%**, ho├án to├án ─æ├íp ß╗⌐ng y├¬u cß║ºu vß║¡n h├ánh thß╗▒c tß║┐.

---

## 16. Dashboard ph├ón t├¡ch v├á kiß╗âm tra trß╗▒c quan

Dß╗▒ ├ín trang bß╗ï mß╗Öt ß╗⌐ng dß╗Ñng Dashboard trß╗▒c quan h├│a chuy├¬n s├óu bß║▒ng Streamlit ([src/dashboard/app_dashboard_tienxuly.py](file:///d:/THUCTAP_VICOMSAT/src/dashboard/app_dashboard_tienxuly.py)) c├╣ng module nß║íp dß╗» liß╗çu ─æa nguß╗ôn ([src/dashboard/dashboard_data.py](file:///d:/THUCTAP_VICOMSAT/src/dashboard/dashboard_data.py)).

### 16.1. Mß╗Ñc ─æ├¡ch sß╗¡ dß╗Ñng Dashboard
- Dashboard l├á **c├┤ng cß╗Ñ R&D nß╗Öi bß╗Ö** d├ánh cho kß╗╣ s╞░ v├á chuy├¬n vi├¬n kiß╗âm tra trß╗▒c quan c├íc chuyß║┐n ─æi thß╗▒c tß║┐.
- Dashboard **kh├┤ng phß║úi** l├á giao diß╗çn cho ng╞░ß╗¥i d├╣ng cuß╗æi v├á **kh├┤ng ─æ╞░a v├áo Docker Image API** ─æß╗â ─æß║úm bß║úo container nhß║╣ nhß║Ñt.
- **Hß╗ù trß╗ú 2 nguß╗ôn dß╗» liß╗çu lß╗¢n**:
  1. **Tß║¡p 9 xe thß╗▒c tß║┐ ─æß║ºy ─æß╗º (`fulltt`)**: Dß╗» liß╗çu h├ánh tr├¼nh thß╗▒c tß║┐ d├ái hß║ín cß╗ºa 9 xe vß║¡n tß║úi (`24H-04650`, `29E-45520`, `29E-45560`, `29E-51878`, `29H-41394`, `29H75028`, `35H-09245`, `90H-03494`, `92H-03625`).
  2. **Bß╗Ö 5 xe `CarFuelHistory`**: `Car 1`, `Car 2`, `Car 3`, `Car 4`, `Car 5` vß╗¢i ─æß║ºy ─æß╗º c├íc ph├ón ─æoß║ín h├ánh tr├¼nh ─æa dß║íng.
- Dashboard hiß╗ân thß╗ï ─æß╗ông thß╗¥i:
  - **─É╞░ß╗¥ng m├áu ─æß╗Å**: Dß╗» liß╗çu th├┤ tß╗½ cß║úm biß║┐n (`RawFuel`).
  - **─É╞░ß╗¥ng m├áu t├¡m**: Dß╗» liß╗çu ─æ├ú khß╗¡ nhiß╗àu l├ám m╞░ß╗út (`AI Smooth-Tracking`).
  - **Biß╗âu ─æß╗ô vß║¡n tß╗æc v├á ─æß╗Ö lß╗çch chuß║⌐n**: Theo d├╡i ─æß╗ông bß╗Ö trß║íng th├íi xe.
  - **Bß║úng Data Inspector**: So s├ính tß╗½ng d├▓ng dß╗» liß╗çu v├á xem l├╜ do ra quyß║┐t ─æß╗ïnh (`QualityFlag`).
- **Kh├│a cß║Ñu h├¼nh Q/R**: Dashboard kh├┤ng cho ph├⌐p can thiß╗çp chß╗ënh sß╗¡a tham sß╗æ Q/R trß╗▒c tiß║┐p tr├¬n giao diß╗çn nhß║▒m ─æß║úm bß║úo kß║┐t quß║ú kiß╗âm thß╗¡ lu├┤n lu├┤n t├íi lß║¡p ─æ╞░ß╗úc (Reproducibility).

### 16.2. Vß╗ï tr├¡ ch├¿n ß║únh giao diß╗çn Dashboard
> *Gß╗úi ├╜ bß╗ò sung ß║únh chß╗Ñp thß╗▒c tß║┐*: Ch├¿n ß║únh chß╗Ñp giao diß╗çn Streamlit tß║íi ─æ╞░ß╗¥ng dß║½n `docs/images/dashboard_overview.png` ─æß╗â minh hß╗ìa r├╡ n├⌐t trß╗▒c quan ─æ╞░ß╗¥ng lß╗ìc m├áu t├¡m b├ím s├ít mß╗⌐c nhi├¬n liß╗çu khi xe chß║íy.

---

## 17. H╞░ß╗¢ng dß║½n c├ái ─æß║╖t v├á Khß╗ƒi chß║íy cß╗Ñc bß╗Ö (Local Setup)

### 17.1. Y├¬u cß║ºu m├┤i tr╞░ß╗¥ng
- Hß╗ç ─æiß╗üu h├ánh: Windows 10/11, Linux (Ubuntu 20.04+), hoß║╖c macOS.
- Python: Phi├¬n bß║ún **Python 3.11** (hoß║╖c 3.10).

### 17.2. C├íc b╞░ß╗¢c c├ái ─æß║╖t chi tiß║┐t

```powershell
# 1. Di chuyß╗ân v├áo th╞░ mß╗Ñc dß╗▒ ├ín
cd THUCTAP-VICOMSAT-D1

# 2. Tß║ío m├┤i tr╞░ß╗¥ng ß║úo c├ích ly
python -m venv .venv

# 3. K├¡ch hoß║ít m├┤i tr╞░ß╗¥ng ß║úo
# Tr├¬n Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Tr├¬n Linux / macOS:
# source .venv/bin/activate

# 4. C├ái ─æß║╖t c├íc th╞░ viß╗çn phß╗Ñ thuß╗Öc
pip install --upgrade pip
pip install -r requirements.txt
```

### 17.3. C├íc lß╗çnh vß║¡n h├ánh hß╗ç thß╗æng

- **Khß╗ƒi chß║íy Microservice REST API**:
  ```powershell
  python -m uvicorn src.service.api:app --host 0.0.0.0 --port 8000 --reload
  ```
  Truy cß║¡p t├ái liß╗çu t╞░╞íng t├íc Swagger UI tß║íi: `http://localhost:8000/docs`

- **Khß╗ƒi chß║íy Dashboard kiß╗âm tra t├¡n hiß╗çu**:
  ```powershell
  streamlit run src/dashboard/app_dashboard_tienxuly.py
  ```

- **Chß║íy giß║ú lß║¡p xe ph├ít dß╗» liß╗çu trß╗▒c tiß║┐p (Simulation)**:
  ```powershell
  python simulate_live_car.py
  ```

- **Chß║íy to├án bß╗Ö bß╗Ö kiß╗âm thß╗¡ tß╗▒ ─æß╗Öng**:
  ```powershell
  python -m pytest tests -q
  ```

- **Xuß║Ñt b├ío c├ío ─æ├ính gi├í KPI**:
  ```powershell
  python scripts/evaluate_smooth_tracking.py
  ```

---

## 18. H╞░ß╗¢ng dß║½n ─æ├│ng g├│i v├á Triß╗ân khai Docker

Hß╗ç thß╗æng ─æ╞░ß╗úc ─æ├│ng g├│i bß║▒ng Docker tß╗æi ╞░u theo chuß║⌐n Microservice d├ánh ri├¬ng cho API.

### 18.1. C├íc biß║┐n m├┤i tr╞░ß╗¥ng hß╗ù trß╗ú cß║Ñu h├¼nh

| T├¬n biß║┐n | Mß║╖c ─æß╗ïnh | ├¥ ngh─⌐a |
| :--- | :--- | :--- |
| `PORT` | `8000` | Cß╗òng dß╗ïch vß╗Ñ lß║»ng nghe b├¬n trong container. |
| `DATABASE_URL` | `sqlite:////app/fuel_data/fuel_records.db` | ─É╞░ß╗¥ng dß║½n CSDL SQLite l╞░u lß╗ïch sß╗¡. |
| `REQUIRE_API_KEY` | `false` | Bß║¡t/tß║»t chß║┐ ─æß╗Ö bß║úo mß║¡t y├¬u cß║ºu API Key ß╗ƒ Header. |
| `API_KEY` | `vcomsat_secret_key_2026` | M├ú kh├│a bß║úo mß║¡t nß║┐u bß║¡t kiß╗âm thß╗▒c. |
| `STATE_BACKEND` | `memory` | C╞í chß║┐ l╞░u trß╗» state: `memory` (trong RAM) hoß║╖c `redis`. |
| `REDIS_URL` | `redis://redis:6379/0` | ─Éß╗ïa chß╗ë m├íy chß╗º Redis khi d├╣ng backend Redis. |
| `STATE_TTL_SECONDS` | `259200` | Thß╗¥i gian hß║┐t hß║ín giß║úi ph├│ng state xe (3 ng├áy). |

### 18.2. C├íc thao t├íc vß║¡n h├ánh Docker

- **Build Docker image**:
  ```powershell
  docker compose build fuel-api
  ```

- **Khß╗ƒi chß║íy API ─æß╗Öc lß║¡p (Mß║╖c ─æß╗ïnh RAM State)**:
  ```powershell
  docker compose up -d fuel-api
  ```

- **Khß╗ƒi chß║íy to├án bß╗Ö hß╗ç sinh th├íi k├¿m Redis cluster**:
  ```powershell
  docker compose --profile redis up -d
  ```

- **Xem log hß╗ç thß╗æng theo thß╗¥i gian thß╗▒c**:
  ```powershell
  docker compose logs -f fuel-api
  ```

- **Kiß╗âm tra sß╗⌐c khß╗Åe container**:
  ```powershell
  curl http://localhost:8000/api/v1/health
  ```

- **Dß╗½ng dß╗ïch vß╗Ñ**:
  ```powershell
  docker compose down
  ```

---

## 19. Danh mß╗Ñc Cß║Ñu h├¼nh hß╗ç thß╗æng (Configuration Reference)

To├án bß╗Ö tham sß╗æ nghiß╗çp vß╗Ñ ─æ╞░ß╗úc ─æß╗ïnh ngh─⌐a tß║¡p trung tß║íi file [src/core/filters/smooth_tracking/config.py](file:///d:/THUCTAP_VICOMSAT/src/core/filters/smooth_tracking/config.py). Kh├┤ng hardcode gi├í trß╗ï tß║íi bß║Ñt kß╗│ module n├áo kh├íc.

| Nh├│m tham sß╗æ | T├¬n tham sß╗æ | Gi├í trß╗ï chuß║⌐n | ─É╞ín vß╗ï | ├¥ ngh─⌐a |
| :--- | :--- | :---: | :---: | :--- |
| **Dung t├¡ch** | `default_capacity` | `200.0` | L├¡t | Dung t├¡ch mß║╖c ─æß╗ïnh khi xe ch╞░a c├│ bß║úng calib. |
| | `minimum_capacity` | `30.0` | L├¡t | Giß╗¢i hß║ín dung t├¡ch tß╗æi thiß╗âu hß╗úp lß╗ç. |
| **Vß║¡n ─æß╗Öng** | `stopped_speed_kmh` | `0.5` | km/h | D╞░ß╗¢i ng╞░ß╗íng n├áy xem nh╞░ xe ─æ├ú dß╗½ng. |
| | `moving_speed_kmh` | `5.0` | km/h | V╞░ß╗út ng╞░ß╗íng n├áy xem nh╞░ xe ─æang chß║íy chß║»c chß║»n. |
| | `low_motion_radius_meters` | `30.0` | M├⌐t | B├ín k├¡nh tr├┤i dß║ít GPS khi ─æß╗ù. |
| **Nhiß╗àu & Ng╞░ß╗íng** | `jitter_floor` | `0.8` | L├¡t | Ng╞░ß╗íng nhiß╗àu rung tß╗æi thiß╗âu cß╗ºa cß║úm biß║┐n. |
| | `spike_floor` | `3.0` | L├¡t | Ng╞░ß╗íng x├íc ─æß╗ïnh xung nhß╗ìn ─æß╗Öt biß║┐n. |
| | `level_shift_floor` | `10.0` | L├¡t | B╞░ß╗¢c nhß║úy tß╗æi thiß╗âu ─æß╗â k├¡ch hoß║ít x├íc nhß║¡n mß║╖t bß║▒ng. |
| **Hiß╗çp ph╞░╞íng sai**| `parked_r` / `parked_q` | `35.0` / `0.03` | ΓÇö | Hß╗ç sß╗æ lß╗ìc khi xe ─æß╗ù nß╗ò m├íy. |
| | `moving_r` / `moving_q` | `45.0` / `0.08` | ΓÇö | Hß╗ç sß╗æ lß╗ìc khi xe chß║íy ─æ╞░ß╗¥ng tr╞░ß╗¥ng. |
| | `strong_noise_r` / `q` | `250.0` / `0.01` | ΓÇö | Hß╗ç sß╗æ dß║¡p tß║»t dao ─æß╗Öng cß╗▒c mß║ính. |
| | `robust_trend_r` / `q` | `6.0` / `1.50` | ΓÇö | Hß╗ç sß╗æ b├ím s├ít dß╗æc ti├¬u hao thß║¡t. |
| **X├íc nhß║¡n Causal**| `upward_hold_steps` | `4` | ─Éiß╗âm | Sß╗æ ─æiß╗âm cß║ºn ─æß╗â c├┤ng nhß║¡n mß╗⌐c t─âng mß╗¢i. |
| | `downward_confirm_points` | `3` | ─Éiß╗âm | Sß╗æ ─æiß╗âm cß║ºn ─æß╗â c├┤ng nhß║¡n mß╗⌐c sß╗Ñt mß╗¢i. |
| **─Éß╗⌐t qu├úng** | `reset_gap_minutes` | `120.0` | Ph├║t | Thß╗¥i gian ─æß╗⌐t t├¡n hiß╗çu ─æß╗â reset state. |

---

## 20. Giß╗¢i hß║ín kß╗╣ thuß║¡t v├á Rß╗ºi ro vß║¡n h├ánh (Known Limitations & Risks)

Khi tiß║┐p nhß║¡n v├á vß║¡n h├ánh hß╗ç thß╗æng, ─æß╗Öi ng┼⌐ kß╗╣ thuß║¡t doanh nghiß╗çp cß║ºn nß║»m r├╡ c├íc giß╗¢i hß║ín vß║¡t l├╜ sau:

1. **Kh├┤ng c├│ Ground Truth tuyß╗çt ─æß╗æi**: Mß╗⌐c nhi├¬n liß╗çu trong b├¼nh di chuyß╗ân tr├¬n ─æ╞░ß╗¥ng thß╗▒c tß║┐ kh├┤ng thß╗â ─æo ch├¡nh x├íc 100% bß║▒ng phao c╞í hß╗ìc. ─É╞░ß╗¥ng CleanFuel l├á ╞░ß╗¢c l╞░ß╗úng tß╗æi ╞░u to├ín hß╗ìc chß╗⌐ kh├┤ng phß║úi ph├⌐p ─æo thß╗â t├¡ch ph├▓ng th├¡ nghiß╗çm.
2. **Sai sß╗æ do dung t├¡ch ╞░ß╗¢c t├¡nh (`CapacityEst`)**: Nß║┐u mß╗Öt xe b├¼nh 600 L├¡t nh╞░ng bß╗ï cß║Ñu h├¼nh nhß║ºm l├á 100 L├¡t, c├íc ng╞░ß╗íng ph├ít hiß╗çn Spike v├á Shift sß║╜ bß╗ï co nhß╗Å lß║íi, dß║½n tß╗¢i hiß╗çn t╞░ß╗úng phß║ún ß╗⌐ng qu├í nhß║íy vß╗¢i dao ─æß╗Öng nhß╗Å.
3. **Hiß╗çn t╞░ß╗úng ─æß╗ù xe tr├¬n dß╗æc d├ái hß║ín**: Nß║┐u xe ─æß╗ù tr├¬n mß╗Öt con dß╗æc nghi├¬ng trong suß╗æt 3 tiß║┐ng, phao nhi├¬n liß╗çu sß║╜ lß╗çch cß╗æ ─æß╗ïnh trong suß╗æt 3 tiß║┐ng ─æ├│. V├¼ hß╗ç thß╗æng kh├┤ng c├│ cß║úm biß║┐n ─æo g├│c nghi├¬ng th├ón xe (Inclinometer), bß╗Ö lß╗ìc sau 4 ─æiß╗âm x├íc nhß║¡n sß║╜ buß╗Öc phß║úi chß║Ñp nhß║¡n mß║╖t bß║▒ng nghi├¬ng n├áy.
4. **─Éß╗Ö trß╗à x├íc nhß║¡n 2ΓÇô4 ─æiß╗âm l├á bß║»t buß╗Öc**: Khi c├│ sß╗▒ kiß╗çn nß║íp hoß║╖c r├║t dß║ºu thß║¡t, hß╗ç thß╗æng kh├┤ng thß╗â k├⌐o ─æ╞░ß╗¥ng t├¡m nhß║úy ngay tß║íi gi├óy ─æß║ºu ti├¬n m├á cß║ºn 4ΓÇô8 ph├║t ─æß╗â khß║│ng ─æß╗ïnh ─æ├│ kh├┤ng phß║úi s├│ng s├ính giß║ú.
5. **Cß║úm biß║┐n bß╗ï lß╗ùi kß║╣t phao**: Nß║┐u phao c╞í hß╗ìc bß╗ï kß║╣t cß╗⌐ng ß╗ƒ l╞░ng chß╗½ng b├¼nh, t├¡n hiß╗çu ─æiß╗çn gß╗¡i vß╗ü l├á mß╗Öt ─æ╞░ß╗¥ng thß║│ng tß║»p ho├án hß║úo. Thuß║¡t to├ín lß╗ìc t├¡n hiß╗çu sß║╜ coi ─æ├óy l├á trß║íng th├íi ß╗òn ─æß╗ïnh t─⌐nh v├á kh├┤ng thß╗â ph├ít hiß╗çn lß╗ùi kß║╣t c╞í kh├¡ nß║┐u thiß║┐u t├¡n hiß╗çu ─æß╗æi so├ít l╞░u l╞░ß╗úng ti├¬u thß╗Ñ.

---

## 21. Lß╗Ö tr├¼nh n├óng cß║Ñp v├á Checklist nghiß╗çm thu b├án giao

### 21.1. Lß╗Ö tr├¼nh ─æß╗ü xuß║Ñt cho c├íc giai ─æoß║ín tiß║┐p theo
- [ ] **T├¡ch hß╗úp bß║úng Calib ─æa ─æiß╗âm (Tank Calibration Table)**: Bß╗ò sung module chuyß╗ân ─æß╗òi trß╗▒c tiß║┐p tß╗½ gi├í trß╗ï ─æiß╗çn ├íp/tß║ºn sß╗æ cß║úm biß║┐n sang L├¡t theo bß║úng dung t├¡ch thß╗▒c tß║┐ cß╗ºa tß╗½ng biß╗ân sß╗æ xe.
- [ ] **Bß╗ò sung k├¬nh t├¡n hiß╗çu gia tß╗æc / ─Éß╗Ö nghi├¬ng (IMU 3 trß╗Ñc)**: Nß║┐u thiß║┐t bß╗ï phß║ºn cß╗⌐ng n├óng cß║Ñp c├│ th├¬m cß║úm biß║┐n gia tß╗æc, thuß║¡t to├ín sß║╜ b├╣ trß╗½ ─æß╗Ö nghi├¬ng dß╗æc tß╗⌐c thß╗¥i m├á kh├┤ng cß║ºn chß╗¥ trß╗à x├íc nhß║¡n.
- [ ] **Mß╗ƒ rß╗Öng Message Queue ph├ón t├ín**: T├¡ch hß╗úp Apache Kafka hoß║╖c RabbitMQ vß╗¢i c╞í chß║┐ **Partition by VehicleID** ─æß╗â mß╗ƒ rß╗Öng quy m├┤ phß╗Ñc vß╗Ñ l├¬n h├áng chß╗Ñc ngh├¼n ph╞░╞íng tiß╗çn ─æß╗ông thß╗¥i.

### 21.2. Checklist nghiß╗çm thu b├án giao doanh nghiß╗çp

Doanh nghiß╗çp thß╗▒c hiß╗çn ─æß╗æi so├ít theo bß║úng kiß╗âm nghiß╗çm thu kß╗╣ thuß║¡t d╞░ß╗¢i ─æ├óy:

- [x] **M├ú nguß╗ôn sß║ích v├á module h├│a**: To├án bß╗Ö thuß║¡t to├ín ─æ╞░ß╗¥ng t├¡m t├ích biß╗çt trong `src/core/filters/smooth_tracking/`.
- [x] **Bß╗Ö kiß╗âm thß╗¡ v╞░ß╗út qua**: 83/83 unit/regression tests chß║íy pass th├ánh c├┤ng.
- [x] **Bß╗Ö kiß╗âm thß╗¡ Golden Segments**: 8/8 ─æoß║ín dß╗» liß╗çu thß╗▒c tß║┐ ─æ╞░ß╗úc ─æ╞░a v├áo kiß╗âm ─æß╗ïnh hß╗ôi quy tß╗▒ ─æß╗Öng.
- [x] **T├¡nh n─âng an to├án ─æa luß╗ông**: Kiß╗âm thß╗¡ `test_concurrent_streaming.py` chß╗⌐ng minh kh├┤ng c├│ Race Condition giß╗»a c├íc xe.
- [x] **Hß╗ù trß╗ú chß║íy Docker**: C├│ sß║╡n Dockerfile ─æa tß║ºng v├á docker-compose.yml khß╗ƒi chß║íy tß╗⌐c thß╗¥i.
- [x] **Health Check Endpoint**: Kiß╗âm tra hoß║ít ─æß╗Öng tß║íi `/api/v1/health`.
- [x] **Hß╗ù trß╗ú State linh hoß║ít**: Chuyß╗ân ─æß╗òi m╞░ß╗út m├á giß╗»a RAM v├á Redis bß║▒ng biß║┐n m├┤i tr╞░ß╗¥ng `STATE_BACKEND`.
- [x] **─Éß║ºy ─æß╗º t├ái liß╗çu t├¡ch hß╗úp**: C├│ t├ái liß╗çu h╞░ß╗¢ng dß║½n v├á m├ú mß║½u cho Python, cURL, C# .NET.

---

## 22. H╞░ß╗¢ng dß║½n quß║ún l├╜ v├á Bß╗ò sung h├¼nh ß║únh trong t├ái liß╗çu (Visual Assets & Guidelines)

Nhß║▒m ─æß║úm bß║úo t├ái liß╗çu b├án giao ─æß║ít t├¡nh trß╗▒c quan cao nhß║Ñt cho ─æß╗æi t├íc doanh nghiß╗çp v├á hß╗Öi ─æß╗ông ─æ├ính gi├í, d╞░ß╗¢i ─æ├óy l├á danh mß╗Ñc chi tiß║┐t c├íc h├¼nh ß║únh ─æ├ú t├¡ch hß╗úp v├á c├íc vß╗ï tr├¡ khuyß║┐n nghß╗ï bß╗ò sung ß║únh minh hß╗ìa:

### 22.1. Danh mß╗Ñc h├¼nh ß║únh hiß╗çn c├│ trong t├ái liß╗çu

| STT | Vß╗ï tr├¡ trong README | ─É╞░ß╗¥ng dß║½n file ß║únh | ─Éß╗ïnh dß║íng | Nß╗Öi dung m├┤ tß║ú kß╗╣ thuß║¡t |
|:---:|:---|:---|:---:|:---|
| 1 | **Mß╗Ñc 1.2** (Giß╗¢i thiß╗çu b├ái to├ín) | `docs/images/filter_comparison_visual.png` | PNG | So s├ính trß╗▒c quan dß╗» liß╗çu th├┤ (`RawFuel`) v├á c├íc ─æ╞░ß╗¥ng lß╗ìc l├ám m╞░ß╗út, l├ám nß╗òi bß║¡t ─æ╞░ß╗¥ng t├¡m th├¡ch ß╗⌐ng (`CleanFuel`). |
| 2 | **Mß╗Ñc 8.1** (M├┤ h├¼nh AI) | `docs/images/test_held_out_confusion_matrix.svg` | Vector SVG | Ma trß║¡n nhß║ºm lß║½n 5x5 tr├¬n tß║¡p kiß╗âm thß╗¡ ─æß╗Öc lß║¡p 3 xe Unseen (23,486 mß║½u, Accuracy 97.33%, kh├┤ng vß╗í hß║ít). |
| 3 | **Mß╗Ñc 8.2** (M├┤ h├¼nh AI) | `docs/images/cm_Car_5.svg` | Vector SVG | Ma trß║¡n nhß║ºm lß║½n chi tiß║┐t cß╗ºa xe ─æß║íi diß╗çn `Car 5` (13,698 mß║½u, Accuracy 95.58%). |
| 4 | **Mß╗Ñc 9.3** (Thuß║¡t to├ín Smooth-Tracking) | `docs/images/adaptive_kalman_behavior.png` | PNG | Biß╗âu ─æß╗ô th├¡ch ß╗⌐ng ─æß╗Öng cß╗ºa c├íc hß╗ç sß╗æ Kalman ($Q$, $R$, Kalman Gain) theo c├íc trß║íng th├íi vß║¡n h├ánh. |

---

### 22.2. C├íc ─æoß║ín cß║ºn bß╗ò sung ß║únh thß╗▒c tß║┐ v├á H╞░ß╗¢ng dß║½n thß╗▒c hiß╗çn

Khi chß╗Ñp ß║únh m├án h├¼nh tß╗½ hß╗ç thß╗æng ─æang chß║íy cß╗Ñc bß╗Ö ─æß╗â bß╗ò sung v├áo b├ío c├ío nghiß╗çm thu, khuyß║┐n nghß╗ï ch├¿n v├áo c├íc mß╗Ñc sau:

#### 1. Mß╗Ñc 16.2 ΓÇö Giao diß╗çn Dashboard R&D Streamlit
- **File ─æß╗ü xuß║Ñt**: `docs/images/dashboard_overview.png`
- **Nß╗Öi dung cß║ºn chß╗Ñp**:
  - Khß╗ƒi chß║íy Dashboard: `streamlit run src/dashboard/app_dashboard_tienxuly.py`
  - Chß╗ìn xe `90H-03494` hoß║╖c `92H-03625` tß╗½ sidebar.
  - Chß╗Ñp m├án h├¼nh thß╗â hiß╗çn ─æß╗ông thß╗¥i: Biß╗âu ─æß╗ô ─æ╞░ß╗¥ng ─æß╗Å `RawFuel` dß║¡p dß╗ünh v├á ─æ╞░ß╗¥ng t├¡m `CleanFuel` m╞░ß╗út m├á, biß╗âu ─æß╗ô vß║¡n tß╗æc/─æß╗Ö lß╗çch chuß║⌐n b├¬n d╞░ß╗¢i, v├á bß║úng dß╗» liß╗çu Data Inspector.
- **├¥ ngh─⌐a**: Gi├║p ng╞░ß╗¥i ─æß╗ìc h├¼nh dung ngay lß║¡p tß╗⌐c giao diß╗çn l├ám viß╗çc trß╗▒c quan cß╗ºa c├íc kß╗╣ s╞░ ph├ón t├¡ch dß╗» liß╗çu.

#### 2. Mß╗Ñc 13.1 ΓÇö T├ái liß╗çu t╞░╞íng t├íc REST API Swagger UI
- **File ─æß╗ü xuß║Ñt**: `docs/images/api_swagger_docs.png`
- **Nß╗Öi dung cß║ºn chß╗Ñp**:
  - Khß╗ƒi chß║íy API: `uvicorn src.service.api:app --reload`
  - Truy cß║¡p tr├¼nh duyß╗çt tß║íi `http://localhost:8000/docs`
  - Chß╗Ñp danh mß╗Ñc c├íc endpoints: `POST /api/v1/fuel/clean-point`, `POST /api/v1/fuel/clean-batch`, `GET /api/v1/health`.
- **├¥ ngh─⌐a**: Chß╗⌐ng minh t├¡nh sß║╡n s├áng triß╗ân khai Microservice theo chuß║⌐n OpenAPI/Swagger cho ─æß╗Öi ng┼⌐ IT doanh nghiß╗çp.

#### 3. Mß╗Ñc 10.3 ΓÇö Minh hß╗ìa chi tiß║┐t tr╞░ß╗¥ng hß╗úp triß╗çt ti├¬u ─æ├íy sß╗Ñt chß╗» U
- **File ─æß╗ü xuß║Ñt**: `docs/images/case_valley_u_recovery.png` (hoß║╖c SVG)
- **Nß╗Öi dung cß║ºn chß╗Ñp**:
  - Zoom cß║¡n cß║únh ─æoß║ín t├¡n hiß╗çu tß╗½ ph├║t thß╗⌐ 0 ─æß║┐n ph├║t thß╗⌐ 30 cß╗ºa mß╗Öt chuyß║┐n ─æi c├│ hiß╗çn t╞░ß╗úng phanh gß║Ñp / leo dß╗æc.
  - Thß╗â hiß╗çn r├╡: T├¡n hiß╗çu ─æo th├┤ bß╗ï tß╗Ñt s├óu dß║íng ─æ├íy chß╗» U nh╞░ng ─æ╞░ß╗¥ng t├¡m `CleanFuel` ─æ╞░ß╗úc giß╗» nguy├¬n nß║▒m ngang (trß║íng th├íi `VALLEY_HOLD`) v├á sau ─æ├│ phß╗Ñc hß╗ôi m╞░ß╗út m├á (`RECOVERY_SMOOTH`).
- **├¥ ngh─⌐a**: Bß║▒ng chß╗⌐ng kß╗╣ thuß║¡t trß╗▒c quan chß╗⌐ng minh thuß║¡t to├ín loß║íi bß╗Å 100% b├ío ─æß╗Öng giß║ú r├║t trß╗Öm nhi├¬n liß╗çu khi xe phanh.

#### 4. Mß╗Ñc 9.5 ΓÇö Minh hß╗ìa ph├ón biß╗çt gai nhß╗ìn Spike ─æ╞ín lß║╗ vs B╞░ß╗¢c nhß║úy mß╗⌐c t─âng
- **File ─æß╗ü xuß║Ñt**: `docs/images/case_spike_vs_upward_shift.png`
- **Nß╗Öi dung cß║ºn chß╗Ñp**:
  - ─Éß║╖t cß║ính nhau 2 t├¼nh huß╗æng:
    1. Mß╗Öt xung gai nhß╗ìn (Spike) t─âng vß╗ìt rß╗ôi r╞íi xuß╗æng ngay trong 1 chu kß╗│ $\rightarrow$ ─æ╞░ß╗¥ng t├¡m phß╗¢t lß╗¥ ho├án to├án (`SPIKE_SUPPRESS`).
    2. Mß╗Öt b╞░ß╗¢c nhß║úy t─âng bß╗ün vß╗»ng duy tr├¼ qua 4 nhß╗ïp $\rightarrow$ ─æ╞░ß╗¥ng t├¡m ─æ╞░ß╗úc k├⌐o l├¬n mß║╖t bß║▒ng mß╗¢i sau khi x├íc nhß║¡n ─æß╗º bß║▒ng chß╗⌐ng (`UPWARD_HOLD_TRACKED`).
- **├¥ ngh─⌐a**: Khß║│ng ─æß╗ïnh ─æß╗Ö tin cß║¡y v├á t├¡nh an to├án cao cß╗ºa c╞í chß║┐ x├íc nhß║¡n ─æa nhß╗ïp Causal.

\n\n## 18. Kiểm thử và trạng thái regression

Chạy suite chính thức:

`powershell
.\.venv\Scripts\python.exe -m pytest -q tests
`

Chạy nhóm trọng yếu đã xác minh sau thay đổi capacity/OperationalGuard:

`powershell
.\.venv\Scripts\python.exe -m pytest -q 
  tests/test_smooth_tracking_noise_symmetry.py 
  tests/test_dashboard_topic1.py 
  tests/test_capacity_initialization.py 
  tests/test_motion_quality_context.py 
  tests/test_purple_service_unification.py 
  tests/test_topic1_api_contract.py 
  tests/test_concurrent_streaming.py
`

Kết quả gần nhất:

`	ext
Nhóm trọng yếu: 43 passed
Toàn bộ tests/: 110 passed, 14 failed
`

Các failure đang chờ review:

1. Một test export history gọi loat(None) khi điểm chưa tạo được CleanFuel.
2. Một số golden fixture truyền capacity_est_liters=200 trong khi RawFuel thực tế trên 400–500 L. Với semantics mới, đây là capacity KNOWN/REQUEST sai và physical clamp tạo kết quả 210 L.
3. Một số snapshot U/GPS lệch nhỏ sau OperationalGuard mới.

Không cập nhật golden snapshot cho tới khi xác minh expected cũ hay output mới hợp lý hơn.

## 19. Giới hạn đã biết

1. **Causal ambiguity:** Một mức thấp kéo dài có thể là baseline thật hoặc sensor excursion. Không có future/ACC/IMU/flow meter thì không thể phân biệt tuyệt đối tại điểm đầu tiên.
2. **Deep dropout policy:** Drop tức thời từ 70% baseline trở lên được giữ cho tới rebound hoặc reset. Đây là lựa chọn an toàn cho sensor-floor dropout nhưng có thể làm chậm một physical shift cực lớn thật sự.
3. **Segment reset trong API:** Schema nhận segment_id, nhưng StreamingStateManager hiện bỏ qua trường này. Dashboard vẫn reset đúng theo segment.
4. **Time-gap reset:** Giá trị code hiện tại đã được cập nhật là 30 phút (trước đây là 120 phút).
5. **Redis operational state:** Serializer hiện lưu Kalman và history cơ bản nhưng chưa lưu đầy đủ các field excursion/recovery mới.
6. **Capacity sai từ caller:** Request capacity hợp lệ về kiểu dữ liệu được coi là KNOWN. Nếu giá trị vật lý sai, clamp và threshold cũng sai.
7. **Không có Ground Truth tuyệt đối:** CleanFuel là ước lượng tín hiệu, không phải phép đo thể tích chuẩn phòng thí nghiệm.

## 20. Quy tắc đóng góp

- Không retrain RF hoặc sửa Ground Truth trong một thay đổi operational nếu chưa có yêu cầu và review riêng.
- Mọi thay đổi filter phải có test causal và replay dữ liệu liên quan.
- Không cập nhật snapshot chỉ để làm test xanh.
- Ghi rõ thay đổi contract/state schema và hướng dẫn reset state khi deploy.\n