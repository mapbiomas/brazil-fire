"""
Interface Language (UI)
MapBiomas Fire Monitor Pipeline

All UI-facing strings must come from class L below.
Tech messages (print/log/raise) must be in plain English, WITHOUT using L.

To add a new language:
  1. Add a `STRINGS_XX` dict below
  2. Add the locale code to `SUPPORTED_LOCALES`
  3. Call `L.load_locale('xx')` at startup
"""


class L:
    """Class attributes resolved at runtime — change values and the whole UI reflects."""

    # Default language is English
    # ── General ───────────────────────────────────────────
    LOADING = "Loading..."
    PROCESSING = "Processing..."
    DELETING = "Deleting..."
    SAVING = "Saving..."
    SYNCING = "Syncing..."
    SEARCHING = "Searching..."
    DONE = "Done"
    ERROR = "Error"
    SUCCESS = "Success"
    WARNING = "Warning"
    INFO = "Info"
    ALL = "All"
    ALL_F = "All"
    NONE = "None"
    BACK = "Back"
    OK = "OK"
    CONFIRM = "Confirm"
    CANCEL = "Cancel"
    SAVE = "Save"
    DELETE = "Delete"
    SYNC = "Sync"
    SEARCH = "Search"
    CLEAR = "Clear"
    REFRESH = "Refresh"
    APPLY = "Apply"
    CLOSE = "Close"
    MODEL = "Model"
    REGION = "Region"
    YEAR = "Year"
    PERIOD = "Period"
    TASK = "Task"
    TASK_NAME = "Task Name"
    STATUS = "Status"
    PROGRESS = "Progress"
    DESCRIPTION = "Description"
    NAME = "Name"
    DATE = "Date"
    ID = "ID"

    # ── Empty states ─────────────────────────────────────
    NO_DATA = "No data available."
    NO_RESULTS = "No results found."
    NO_TASKS = "No pending tasks."
    NO_TASKS_PUBLISH = "No tasks ready to publish."
    NO_TASKS_DONE = "No finished tasks yet."
    NO_TILES_GCS = "No tiles in GCS."
    NO_MAP = "Could not generate map. Check GEE connection."
    NO_SELECTION = "No option selected."
    NO_SAMPLES = "No samples found with this filter."
    NO_COGS = "No COGs found in the GCS repository."

    # ── Widgets / Buttons M5 ─────────────────────────────
    ADD_BATCH = "Add Batch to Workplan"
    REFRESH_VIEW = "Refresh View"
    LOAD_TO_QUEUE = "Load to Workplan"
    CLEAR_TEMP_TASKS = "Clear Temporary Tasks"
    SAVE_TASK_GCS = "Save Task GCS"
    EXCLUDE_TASK_GCS = "Exclude Task GCS"
    DELETE_MODEL = "Delete Model"
    DISCARD_WORKPLAN = "Discard Workplan"
    GLOBAL_ACTIONS = "Global Actions"
    NO_PENDING_JOBS = "No pending jobs to classify."
    ERR_NO_SAMPLES = "Error: No samples selected."
    ERR_NO_BANDS = "Error: No bands selected in the Extraction Matrix."
    ERR_INCOMPATIBLE_ENV = "Incompatible local environment"
    ERR_NO_AVX = "Your CPU does not support AVX/AVX2 instructions required by TensorFlow."
    ERR_RUN_ON_COLAB = "Please: Run this training on Google Colab."
    CARD_SAVED = "Saved \u2713"
    CARD_TEMP = "Temporary"
    CARD_SAVED_PARTIAL = "{s}/{t} Saved"
    BTN_DETAILS_COLLAPSED = "Details \u25bc"
    BTN_DETAILS_EXPANDED = "Details \u25b2"
    NO_METADATA = "Model metadata unavailable (offline or missing metadata.json)."
    DELETE_REGION = "Delete Region"
    DELETE_SELECTED = "Delete Selected"
    DELETE_ALL = "Delete All"
    DELETE_JOB = "Delete All"
    VIEW_TILES = "View Tiles"
    HIDE_TILES = "Hide"
    REFRESH_MAP = "Refresh Map"
    TASK_NAME_PLACEHOLDER = "e.g.: Classify Low Amazon 2025 (Lucas)"

    # ── M5 - Section titles ────────────────────────────
    SECTION_MODEL = "1. Select Model (single):"
    SECTION_REGIONS = "2. Select Regions:"
    SECTION_PERIODS = "3. Select Periods (Year / Year_Month):"
    LOADING_CAMPAIGNS = "Loading campaigns..."
    CAMPAIGN_LABEL = "Campaign:"
    FILTER_LABEL = "Filter:"
    FILTER_PLACEHOLDER = "search by model, region, period..."

    # ── M5 - Region diagnostic ─────────────────────────
    ERR_CONFIG_ASSET_REGIONS = "CONFIG[asset_regions] not defined"
    ERR_ASSET_NOT_FOUND = "Asset NOT FOUND in GEE:"
    ERR_ASSET_EMPTY = "Asset exists but is EMPTY:"
    ERR_ASSET_COLUMN = "Column <code>{prop}</code> not found"
    ERR_GEE_ACCESS = "Error accessing GEE:"
    WARN_REGION_FALLBACK = "Could not auto-populate regions"
    ERR_RENDER_PENDING = "Error rendering pending:"
    COLUMN_USED = "Column used:"

    # ── M5 - Validation ───────────────────────────────
    ERR_NO_TASK_NAME = "Attention: You must assign a Task Name."
    ERR_NO_SELECTION = "Attention: Select 1 Model and at least 1 Region and 1 Period."

    # ── M5 - Feedback messages ─────────────────────────
    SAVING_TO_GCS = "Saving to GCS..."
    MSG_NO_TASKS_ADDED = "No tasks were added."
    MSG_TASKS_ADDED = "Success: {n} task(s) added."
    FAIL_SAVE_LOCAL = "failed to save local file (lock busy)"
    FAIL_GCS_UPLOAD = "{n} failed on GCS"
    MSG_GCS_OK = "{n} saved to GCS."
    MSG_SKIPPED_QUEUED = "{n} skipped (already in queue)."
    WARN_ALREADY_QUEUED = "Warning: {n} tasks already in queue."
    MSG_LOADED = "{loaded} loaded, {skipped} skipped."
    TAREA_SAVED = "Task saved to GCS for {m}."
    TAREA_DELETED = "Task deleted from GCS: {m}."
    WORKPLAN_DISCARDED = "Workplan discarded: {m} ({n} jobs removed)."
    QUEUE_CLEARED = "Queue cleared (GCS removed)."
    SAVED_TO_GCS = "{n} job(s) saved to GCS."
    ALREADY_SAVED = "{n} job(s) already saved."
    TEMP_JOBS_REMOVED = "{n} temporary job(s) removed."
    NO_TEMP_JOBS = "No temporary jobs to remove."
    REMOVED_FROM_GCS = "{n} job(s) removed from GCS."
    NO_TILES_GCS_ALT = "No tiles in GCS."
    TILES_DELETED = "{n} tile(s) deleted."
    SELECT_TILES_PROMPT = "Select tiles to delete."
    SYNC_FROM_GCS = "{n} job(s) synced from GCS."

    # ── M5 - Tareas section ────────────────────────────
    TAREAS_HEADER = "Tasks in GCS ({n} available)"
    TAREA_SUMMARY = "Model: {m} | Regions: {r} | Periods: {p}"

    # ── M5 - Scale bar ─────────────────────────────────
    SCALE_BAR_WIDTH = "~{w} km wide"

    # ── M5 - Pending debug ─────────────────────────────
    INFO_FILTER_COUNT = "{total} pending | {filtered} after filter"
    GRID_COUNT = "({n} cells)"

    # ── M5 - Map legend ────────────────────────────────
    MAP_LEGEND_COUNTRY = "Country"
    MAP_LEGEND_REGIONS = "Regions"
    MAP_LEGEND_GRID = "Grid cim-world"

    # ── M5 - Model inference errors ────────────────────
    ERR_MODEL_NO_METADATA = "metadata.json not found at {path}"
    ERR_MODEL_NO_BANDS_CONFIG = "Model has no 'bands_config' in metadata."
    ERR_MODEL_NO_BAND_ORDER = "Model has no 'band_order' in metadata. Retrain with the latest M4 (commit 9b95392+)."
    ERR_MODEL_INCONSISTENCY = "Metadata inconsistency: {details}"

    # ── M5 - Jobs details panel ────────────────────────
    JOBS_REGISTERED = "Registered jobs"
    JOBS_SUMMARY = "{total} job(s), {regions} region(s), {periods} period(s)"
    JOB_REGION = "Region"
    JOB_PERIOD = "Period"
    JOB_STATUS = "Status"
    JOB_TASK = "Task"

    # ── M5 - Model metadata panel ──────────────────────
    META_SENSORS = "Sensors"
    META_PERIODICITY = "Periodicity"
    META_BANDS_INPUT = "Bands (input)"
    META_ITERATIONS = "Iterations"
    META_LEARNING_RATE = "Learning rate"
    META_CAMPAIGN = "Campaign"

    # ── M5 Tabs ──────────────────────────────────────────
    TAB_GUIDE = "Guide"
    TAB_REGISTER = "Register"
    TAB_PENDING = "Pending"
    TAB_PUBLISH = "To Publish"
    TAB_MAP = "Map"
    TAB_DONE = "Finished"

    # ── M6 Tabs ──────────────────────────────────────────
    TAB_ANALYTICS = "Analytics"
    TAB_M6_COVERAGE = "Coverage"
    GUIDE_M6_HTML = """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>M6 - Mosaic, Stats & Publication</h3>
        <p>Publishes classified tiles from M5: creates regional mosaics, computes burned area statistics, and uploads to Google Earth Engine.</p>
        <h4>Flow:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_publish}</b> — groups of classified tiles waiting for mosaic.</li>
            <li><b>{tab_done}</b> — published regions with mosaic and stats.</li>
            <li><b>{tab_analytics}</b> — consolidated stats table with download.</li>
            <li><b>{tab_coverage}</b> — coverage map: what's published vs pending.</li>
            <li>Run <code>run_m6_publish()</code> in the notebook to process.</li>
        </ol>
    </div>"""
    DOWNLOAD_TABLE = "Download Table"
    ANALYTICS_FILTER_MODEL = "Model"
    ANALYTICS_FILTER_REGION = "Region"
    ANALYTICS_FILTER_PERIOD = "Period"
    RUN_M6 = "Run M6 Publish"
    REFRESH_M6 = "Refresh M6"
    M6_HEADER_TITLE = "M6 - Mosaic, Stats & Publication"
    M6_LABEL_PERIOD = "Period:"
    M6_GROUPS_PENDING = "{n} groups pending mosaic"
    M6_MOSAIC_OK = "mosaic OK"
    M6_DOWNLOAD_MOSAIC = "OK \u2197"
    M6_PUBLISHED_GROUPS = "{n} published groups"
    M6_BADGE_MOSAIC = "Mosaic"
    M6_BADGE_STATS = "Stats"
    M6_BADGE_GEE = "GEE"
    M6_COL_PCT = "%"
    M6_COL_HA = "ha"
    M6_COL_TILES = "Tiles"
    M6_LEGEND_PUBLISHED = "Published"
    M6_LEGEND_PARTIAL = "Partial"
    M6_LEGEND_CLASSIFIED_ONLY = "Classified only"
    M6_LEGEND_NO_DATA = "No data"
    M6_N_RECORDS = "{n} records"
    M6_DOWNLOAD_LINK = "Download {fname}"
    M6_NO_STATS = "No consolidated stats available. Run M6 publish first."
    M6_NO_MATCHING = "No matching records."
    M6_NO_CLASSIFIED_GROUPS = "No classified groups found."

    GUIDE_M5_HTML = """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:5px;'>M5 - Large Scale Regional Classification</h3>
        <p>Classifies multiple regions (cim-world-1-250000 grid cells) using M4 models.</p>
        <h4>Flow:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_register}</b> — select model + regions + periods.</li>
            <li><b>{tab_pending}</b> — follow classification tile by tile.</li>
            <li><b>{tab_map}</b> — live processing overview.</li>
            <li>Run <code>run_m5_workplan()</code> in the notebook to process.</li>
        </ol>
        <h4>After Classification (M6):</h4>
        <p>Use <b>M6</b> to mosaic tiles, generate regional stats, and upload to GEE.
        Open the M6 UI and run <code>run_m6_publish()</code> in a separate cell.</p>
        <p>Classified jobs (COMPLETED) appear automatically in M6.</p>
    </div>"""

    GUIDE_M1_HTML = """<div style='padding:20px; font-family:sans-serif;'>
        <div style='margin-bottom:15px;padding:10px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px;font-size:13px;'>
            <b>Note:</b> Currently only monthly composites for <b>Sentinel-2</b> with <b>minnbr</b> and <b>minnbr_buffer</b> are fully released. Other sensor/period/mosaic combinations may be experimental.
        </div>
        <h3 style='color:#2c3e50; border-bottom:2px solid #e67e22; padding-bottom:5px;'>Module 1 (M1) — Multi-Sensor GEE Export</h3>
        <p>M1 exports seamless, cloud-free satellite mosaics from Google Earth Engine to Google Cloud Storage (GCS) or GEE Assets.
        Supports LANDSAT 5/7, LANDSAT 8/9, Sentinel-2, HLS, and MODIS with sensor-specific radiometric corrections and cloud masking.</p>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:15px;'>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Sensors &amp; Sources</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>LANDSAT 5/7</b> — historical archive (1984–)</li>
                    <li><b>LANDSAT 8/9</b> — current OLI/TIRS thermal bands</li>
                    <li><b>Sentinel-2</b> — 10m resolution, 5-day revisit</li>
                    <li><b>HLS</b> — harmonized LANDSAT/Sentinel product</li>
                    <li><b>MODIS</b> — daily global coverage (250m–1km)</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Mosaic Methods</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>minnbr</b> — least cloud from NBR ranking</li>
                    <li><b>minndvi</b> — least cloud from NDVI ranking</li>
                    <li><b>median</b> — per-pixel composite median</li>
                    <li><b>minnbr_buffer</b> — minnbr with INPE fire buffer mask</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Workflow</h4>
                <ol style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Select sensor tab (LANDSAT, S2, HLS, MODIS)</li>
                    <li>Pick period (monthly / annual)</li>
                    <li>Choose mosaic method</li>
                    <li>Check date &amp; band cells</li>
                    <li>Click <b>Start Export</b></li>
                </ol>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Technical Info</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Per-sensor radiometric correction</li>
                    <li>Cloud masking via QA_PIXEL, Fmask, CS+</li>
                    <li>INPE fire perimeter buffer mask</li>
                    <li>Output: GeoTIFF chunks in GCS / ImageCollection in GEE</li>
                </ul>
            </div>
        </div>
        <div style='margin-top:15px;padding:10px;background:#fef3e2;border-left:4px solid #e67e22;border-radius:4px;font-size:13px;'>
            <b>Tip:</b> Use <b>Sync Data</b> to refresh the cache. Use <b>Select Pending</b> to auto-check all available dates.
        </div>
    </div>"""

    GUIDE_M2_HTML = """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>Module 2 (M2) — COG Mosaic Assembly</h3>
        <p>M2 assembles the GeoTIFF chunks exported by M1 into full national Cloud-Optimized GeoTIFFs (COGs) using GDAL.</p>
        <h4>Flow:</h4>
        <ol style='line-height:1.6;'>
            <li><b>Check status</b> — OK (COG exists), READY (chunks available), MISS (nothing available)</li>
            <li><b>Select cells</b> — check the bands you want to assemble</li>
            <li>Click <b>Start Assembly</b></li>
            <li>GDAL downloads chunks → builds VRT → converts to COG (LZW) → uploads to GCS</li>
        </ol>
        <h4>Requirements:</h4>
        <ul style='line-height:1.6;'>
            <li>GDAL installed (<code>gdalbuildvrt</code>, <code>gdal_translate</code>)</li>
            <li>M1 must have been run first to produce source chunks</li>
        </ul>
        <div style='margin-top:15px;padding:10px;background:#e8f8ed;border-left:4px solid #27ae60;border-radius:4px;font-size:13px;'>
            <b>Tip:</b> COGs are stored in <code>.../COG/</code> and consumed by M4 (model training) and M5 (classification).
        </div>
    </div>"""

    # ── Map / Grid ───────────────────────────────────────
    LIVE_PROCESSING = "Live Processing"
    CURRENT_TILE = "Current tile"
    COMPLETED = "Completed"
    GRID_REGION = "Region"
    GRID_CELLS = "cim-world cells"

    # ── Tiles / GCS ──────────────────────────────────────
    TILES = "tiles"
    MOSAIC = "mosaic"
    STATS = "stats"
    SELECT_TILES_DELETE = "Select tiles to delete."
    TILES_REMOVED = "tiles removed."

    # ── Task status ──────────────────────────────────────
    STATUS_QUEUED = "In Workplan"
    STATUS_RUNNING = "Running"
    STATUS_COMPLETED = "Completed"
    STATUS_FINISHED = "Finished"
    STATUS_ERROR = "Error"
    STATUS_PUBLISHING = "Publishing"
    STATUS_PUBLISHED = "Published"
    STATUS_SKIPPED = "Skipped"

    # ── M4 - Training ────────────────────────────────────
    MODEL_TRAINER = "Model Trainer"
    M4_HEADER_TITLE = "M4 - Model Trainer"
    SELECTED_OK = "[OK] Selected"
    TRAINING_PROGRESS = "Training:"
    ITERATIONS = "Iterations"
    BATCH_SIZE = "Batch Size"
    LEARNING_RATE = "Learning Rate"
    HIDDEN_LAYERS = "Hidden Layers"
    ACTIVATION = "Activation"
    DROPOUT = "Dropout"
    OPTIMIZER = "Optimizer"
    LOSS_FN = "Loss Function"
    METRICS = "Metrics"
    SAMPLE_SELECTION = "Sample Selection"
    EXTRACTION_MATRIX = "Extraction Matrix"
    MODEL_CONFIG = "Model Configuration"
    GCS_DEST = "GCS Destination"
    USAGE_GUIDE = "Usage Guide"

    GUIDE_M4_HTML = """<div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; background: #fdfdfd; color: #2c3e50; line-height: 1.6;">
        <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">M4 Model Trainer - Usage Guide</h1>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 20px;">
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #3498db; margin-top:0;">Platform Structure</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>{usage_guide}:</b> This orientation and documentation screen.</li>
                    <li><b>{new_training}:</b> Configuration of new experiments, sample and band selection.</li>
                    <li><b>{trainings}:</b> Historical ranking with detailed metrics and model management.</li>
                    <li><b>Canvas:</b> Parallel audit desk to compare multiple models in depth.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #9b59b6; margin-top:0;">Technical Concepts</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>TensorFlow:</b> Google's AI engine for massive mathematical computations.</li>
                    <li><b>DNN (Deep Neural Network):</b> Deep network that mimics human learning.</li>
                    <li><b>Neurons:</b> Units that process signals and activate learning patterns.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #e67e22; margin-top:0;">Hyperparameters (DNN)</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Layers:</b> Network architecture. More layers capture finer details.</li>
                    <li><b>Learning Rate (LR):</b> Controls how fast the model adjusts.</li>
                    <li><b>Epochs:</b> Complete training cycles over the sample set.</li>
                    <li><b>Batch Size:</b> Data blocks processed before each update.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #27ae60; margin-top:0;">Quality Dictionary</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Accuracy:</b> Total percentage of global hits.</li>
                    <li><b>Precision:</b> Fidelity: How much of the marked fire is real? (Avoids false positives).</li>
                    <li><b>Recall:</b> Coverage: How much of the real fire was found? (Avoids omissions).</li>
                    <li><b>F1-Score:</b> Harmonic mean. The best balance between Precision and Recall.</li>
                    <li><b>AI Note:</b> Automatic audit that severely penalizes omissions.</li>
                    <li><b>Human Note:</b> Subjective evaluation (1-5) on the Latent Space.</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 30px; padding: 15px; background: #e8f4fd; border-left: 5px solid #3498db; border-radius: 4px; font-size:14px;">
            <b>[Tip] Auditor's Pro-Tip:</b> Use the <b>Canvas</b> to load an old model (benchmark) and your new model. Compare if the t-SNE 3D class separation has improved or if there are new confusion zones.
        </div>
    </div>"""

    # ── M4 - Canvas ──────────────────────────────────────
    METADATA = "Metadata"
    KPIS = "KPIs"
    CONFUSION = "Confusion"
    HISTORY = "History"
    PROB = "Prob"
    PR_CURVE = "PR-Curve"
    MANAGEMENT = "Management"
    RANKING = "Ranking / Repository"
    SELECTED_CANVAS = "Selected in Canvas"
    SEARCH_REPO = "Search repository..."
    SEARCH_SAMPLES = "Search samples..."
    SYNC_CATALOG = "Sync Catalog (GCS)"
    APPLY_VISIBILITY = "Apply Visibility"

    # ── M1 / M2 - Export / Mosaic ────────────────────────
    SYNC_DATA = "Sync Data"
    SELECT_PENDING = "Select Pending"
    CLEAR_SELECTION = "Clear Selection"
    EXPORT_START = "Start Export"
    MOSAIC_START = "Start Assembly"

    # ── M6 - Post-classification ─────────────────────────
    POST_CLASSIFICATION = "Post-Classification Processor"
    USING_PRESET = "Using Preset Configuration"
    FILTER_START = "Start Filtering"
    EXPORT_TASK_STARTED = "Export task started"
    CONFIG_SUMMARY = "Configuration Summary"

    # ── M7 - Curator ─────────────────────────────────────
    CURATOR = "Curator"
    CURATOR_DESC = "Pre-Official Collection Publication"
    USING_VOTES_PRESET = "Using Preset Votes"
    CURATION_START = "Start Curation"
    EXPORT_ASSET_STARTED = "GEE Asset export started"

    # ── Cache ────────────────────────────────────────────
    CACHE_REMOVED = "Local cache removed"
    CACHE_NOT_FOUND = "Cache not found"

    # ── Misc ─────────────────────────────────────────────
    LOADING_TILES = "Loading tiles..."
    LOADING_INTERFACE = "Loading interface..."
    CLICK_TO_LOAD = "Click to load {sensor} {period} ({mosaic})..."
    LOADING_CACHE = "Loading cache..."
    SYNCING_TASKS = "Syncing GEE tasks..."
    FILTERING = "Filtering..."
    CONFIRM_DELETE = "Confirm Deletion"
    CONFIRM_DELETE_ALL = "Confirm Delete All"
    CANCELED = "Canceled."
    GLOBAL_OPTS_SET = "Global options set"

    # ── Dropdown labels ─────────────────────────────────
    DROP_MODEL = "Model:"
    DROP_REGION = "Region:"
    DROP_YEAR = "Year:"
    DROP_TASK = "Task:"
    DROP_TASK_NAME = "Task Name:"

    # ── M4 - Hyperparams ────────────────────────────────
    HP_EPOCHS = "Epochs"
    HP_PATIENCE = "Patience"
    HP_TEST_SPLIT = "Test Split"
    HP_BALANCE = "Balance"
    HP_AUGMENT = "Data Augmentation"
    HP_ACTIVATION = "Activation Function"
    HP_OPTIMIZER = "Optimizer"
    HP_LOSS = "Loss Function"

    # ── M4 - Extraction ─────────────────────────────────
    EXTRACTION_TITLE = "Extraction Matrix (Multisensor GCS)"
    SAMPLE_SELECT = "Sample Selection"
    AVAILABLE = "Available"
    SELECTED = "Selected"
    ADD = "Add"
    REMOVE = "Remove"
    CAMPAIGN = "Campaign"
    SENSOR = "Sensor"
    BANDS = "Bands"
    STATUS_OK = "OK"
    STATUS_RUN = "Run"
    STATUS_MISS = "Miss"

    # ── M4 - Training ───────────────────────────────────
    TRAINING_ID = "Training ID:"
    SHORTNAME = "Short Name:"
    COMMENTS = "Comments..."
    START_TRAINING = "Start Training"
    CANVAS_TITLE = "Training and Audit Center"
    CANVAS_EMPTY = "Empty canvas"
    CANVAS_HINT = "Browse and select models in the side panel to visualize them here."
    LOSS_ACC = "Loss / Accuracy"
    VIZ_METADATA = "Metadata"
    VIZ_KPIS = "KPIs"
    VIZ_CONFUSION = "Confusion"
    VIZ_HISTORY = "History"
    VIZ_PROB = "Prob"
    VIZ_PR_CURVE = "PR-Curve"
    VIZ_MANAGEMENT = "Management"

    # ── M4 - Repository ─────────────────────────────────
    REPO_TITLE = "Ranking / Repository"
    REPO_SEARCH = "Search repository..."
    REPO_SYNC = "Sync GCS"
    REPO_ALL = "All"
    REPO_CLEAR = "Clear"
    REPO_SYNC_CATALOG = "Sync Catalog (GCS)"
    REPO_SCANNING = "Scanning GCS... Please wait."
    REPO_SCAN_DONE = "Catalog synced!"

    # ── M4 - Tab titles ─────────────────────────────────
    SORT_BY = "Sort by:"
    NEW_TRAINING = "New Training"
    TRAININGS = "Trainings"

    # ── M1 - Export ─────────────────────────────────────
    EXPORT_TITLE = "Collection Export"
    EXPORT_SYNC = "Sync Data"
    EXPORT_SELECT = "Select Pending"
    EXPORT_CLEAR = "Clear Selection"
    EXPORT_START_BTN = "Start Export"
    EXPORT_SENT = "tasks sent."
    EXPORT_NONE_SEL = "No option selected for export."

    M1_HEADER_TITLE = "M1 - Dispatcher"
    M1_HEADER_SUBTITLE = "Multi-sensor export interface"
    M1_TITLE_FULL = "M1 - Mosaic Dispatcher"
    M1_DESCRIPTION = "Multi-sensor interface to dispatch compositions (Assets/GCS) to the cloud."
    LABEL_PROJECT = "Project"
    BTN_SELECT_ROW = "[S]"
    COL_TYPE = "Type"

    # ── M2 - Mosaic ─────────────────────────────────────
    MOSAIC_TITLE = "Mosaic Assembly"
    MOSAIC_SYNC = "Sync Data"
    MOSAIC_SELECT = "Select Pending"
    MOSAIC_CLEAR = "Clear Selection"
    MOSAIC_START_BTN = "Start Assembly"
    MOSAIC_DONE = "Assembly completed successfully"
    M2_HEADER_TITLE = "M2 - Assembler"
    M2_HEADER_SUBTITLE = "Interface for national COG mosaic assembly"
    M2_CONSTRUCTOR_TITLE = "M2 - COG Mosaic Assembler"
    M2_CONSTRUCTOR_DESC = "Interface for converting GCS chunks into national COG mosaics."
    STATUS_READY = "READY"

    # ── M0 - Startup ────────────────────────────────────
    START_COUNTRY = "Country"
    START_BUCKET = "Bucket"
    START_VERSION = "Version"
    START_SENSOR = "Sensor"
    START_PERIODICITY = "Periodicity"
    START_CAMPAIGN = "Campaign"
    START_FOUND = "found"
    START_NOT_FOUND = "not found"
    START_GDAL_OK = "GDAL found and added to PATH"
    START_GDAL_MISSING = "Warning: GDAL utilities not found"
    START_COLAB_HINT = "On Google Colab, run: !apt-get install -y gdal-bin"
    START_WIN_HINT = "On Windows, ensure GDAL is in your PATH"

    # ── M6 - Post-classification ────────────────────────
    M6_TITLE = "Post-Classification Processor"
    M6_PRESET = "Using Preset Configuration"
    M6_DOWNLOAD = "Downloading fragments"
    M6_VRT = "Building VRT"
    M6_COG = "Converting to COG"
    M6_UPLOAD = "Uploaded"
    M6_START = "Starting Filtering"
    M6_EXPORT_OK = "Export task started"
    M6_SUMMARY = "Configuration Summary"

    # ── M7 - Curator ────────────────────────────────────
    M7_TITLE = "Curator"
    M7_DESC = "Pre-Official Collection Publication"
    M7_PRESET = "Using Preset Votes"
    M7_START = "Starting Curation"
    M7_EXPORT_OK = "GEE Asset export started for"
    M7_SUMMARY = "Configuration Summary"

    # ── M3 - Samples ────────────────────────────────────
    M3_TITLE = "M3 - Sample Collection (GEE Toolkit Gateway)"
    M3_SOURCE = "Source code (GitHub)"
    M3_EDITOR = "Direct access (GEE Editor)"
    M3_DOCS = "Documentation and usage guidelines"
    M3_SLIDES = "Tutorial slides (Google Slides)"
    M3_INTRO_LINE1 = "The sample collection stage is carried out exclusively"
    M3_INTRO_LINE2 = "through the JavaScript interface in Google Earth Engine."

    # ── M4 - Analytics / KPIs ──────────────────────────
    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    F1_SCORE = "F1-Score"
    AUTO_RATING = "Auto Rating"
    HUMAN_NOTE = "Your Rating"
    NO_METRICS = "No metrics"
    METRICS = "Metrics"
    SAVE_RATING = "Save"
    COMMENTS_EDIT = "Edit"
    COMMENTS_SAVE = "Save"
    COMMENTS_CANCEL = "Cancel"
    CONFUSION_MATRIX = "Confusion Matrix (%)"
    HISTORICAL_EVOLUTION = "Historical Evolution"
    PROB_DISTRIBUTION = "Probability Distribution"
    CONFIDENCE = "Confidence"
    LATENT_PROJ_2D = "2D Latent Projection"
    COST_LOSS = "Cost (Loss)"
    CLASSIC_METRICS = "Classic Metrics and Static Projections"
    INTERACTIVE_LATENT = "Interactive Latent Space"
    PCA_3D = "PCA 3D"
    TSNE_3D = "t-SNE 3D"
    PCA_3D_INTERACTIVE = "PCA 3D Interactive"
    TSNE_3D_INTERACTIVE = "t-SNE 3D Interactive"
    RETRAIN = "Retrain"

    # ── M4 - Fire class labels ────────────────────────
    FIRE = "Fire"
    NO_FIRE = "No-fire"
    FIRE_CLASS = "Fire"
    NO_FIRE_CLASS = "No-fire"

    # ── M4 - UI Labels ────────────────────────────────
    BASIC_STATS = "Basic Statistics"
    PCA_LATENT = "PCA Latent Space"
    TSNE_LATENT = "t-SNE Latent Space"
    VIZ_OPTIONS = "Visualization Options"
    VIZ_PCA2D = "PCA 2D"
    VIZ_PCA3D_STATIC = "PCA 3D (Static)"
    VIZ_PCA3D_INTERACTIVE = "PCA 3D (Interactive)"
    VIZ_TSNE3D_STATIC = "t-SNE 3D (Static)"
    VIZ_TSNE3D_INTERACTIVE = "t-SNE 3D (Interactive)"
    HYPERPARAMS_SECTION = "Hyperparameters (DNN)"
    LIVE_TRAINING = "Live Training"
    TRAINING_IN_PROGRESS = "Training in progress..."
    LIVE_TSNE_AUDIT = "Live t-SNE Audit (Latent Space)"
    ADD_TO_CANVAS = "Add to Canvas"
    REMOVE_FROM_CANVAS = "Remove from Canvas"
    BTN_CLOSE = "X"
    RELOAD_SAMPLES = "Reload samples list from GCS"
    PLUS = "+ "

    # ── M4 - Analytics card labels ────────────────────
    LAYERS_LABEL = "Layers:"
    LR_ABBR = "LR:"
    SAMPLES_LABEL = "Samples:"
    NO_COMMENTS = "No comments."
    HIDE_ADVANCED = "Hide advanced parameters"
    SHOW_ALL_PARAMS = "Show all parameters"
    TSNE_AXIS_1 = "t-SNE 1"
    TSNE_AXIS_2 = "t-SNE 2"
    TSNE_AXIS_3 = "t-SNE 3"
    ACC_ABBR = "Acc"
    F1_ABBR = "F1"


# ── Locale dictionaries ────────────────────────────────────

STRINGS_ES = {
    # General
    "LOADING": "Cargando...",
    "PROCESSING": "Procesando...",
    "DELETING": "Eliminando...",
    "SAVING": "Guardando...",
    "SYNCING": "Sincronizando...",
    "SYNCING_TASKS": "Sincronizando tareas GEE...",
    "SEARCHING": "Buscando...",
    "DONE": "Hecho",
    "ERROR": "Error",
    "SUCCESS": "Éxito",
    "WARNING": "Atención",
    "INFO": "Información",
    "ALL": "Todos",
    "ALL_F": "Todas",
    "NONE": "Ninguno",
    "BACK": "Volver",
    "OK": "OK",
    "CONFIRM": "Confirmar",
    "CANCEL": "Cancelar",
    "SAVE": "Guardar",
    "DELETE": "Eliminar",
    "SYNC": "Sincronizar",
    "SEARCH": "Buscar",
    "CLEAR": "Limpiar",
    "REFRESH": "Actualizar",
    "APPLY": "Aplicar",
    "CLOSE": "Cerrar",
    "MODEL": "Modelo",
    "REGION": "Región",
    "YEAR": "Año",
    "PERIOD": "Período",
    "TASK": "Tarea",
    "TASK_NAME": "Nombre Tarea",
    "STATUS": "Estado",
    "PROGRESS": "Progreso",
    "DESCRIPTION": "Descripción",
    "NAME": "Nombre",
    "DATE": "Fecha",
    "ID": "ID",
    # Empty states
    "NO_DATA": "No hay datos disponibles.",
    "NO_RESULTS": "No se encontraron resultados.",
    "NO_TASKS": "No hay tareas pendientes.",
    "NO_TASKS_PUBLISH": "Ninguna tarea lista para publicar.",
    "NO_TASKS_DONE": "Ninguna tarea finalizada aún.",
    "NO_TILES_GCS": "Ningún tile en GCS.",
    "NO_MAP": "No se pudo generar el mapa. Verifique conexión GEE.",
    "NO_SELECTION": "Ninguna opción seleccionada.",
    "NO_SAMPLES": "No se encontraron muestras con este filtro.",
    "NO_COGS": "No se encontraron COGs en el repositorio GCS.",
    # M5 Widgets
    "ADD_BATCH": "Agregar Lote al Plan",
    "REFRESH_VIEW": "Actualizar Vista",
    "LOAD_TO_QUEUE": "Cargar al Plan",
    "CLEAR_TEMP_TASKS": "Limpiar Tareas Temporales",
    "SAVE_TASK_GCS": "Guardar Tarea GCS",
    "EXCLUDE_TASK_GCS": "Excluir Tarea GCS",
    "DELETE_MODEL": "Eliminar Modelo",
    "DISCARD_WORKPLAN": "Descartar Plan de Trabajo",
    "GLOBAL_ACTIONS": "Acciones Globales",
    "NO_PENDING_JOBS": "No hay trabajos pendientes para clasificar.",
        "ERR_NO_SAMPLES": "Error: No hay muestras seleccionadas.",
        "ERR_NO_BANDS": "Error: No hay bandas seleccionadas en la Matriz de Extracción.",
        "ERR_INCOMPATIBLE_ENV": "Entorno local incompatible",
        "ERR_NO_AVX": "Tu CPU no soporta las instrucciones AVX/AVX2 requeridas por TensorFlow.",
        "ERR_RUN_ON_COLAB": "Por favor: Ejecuta este entrenamiento en Google Colab.",
    "CARD_SAVED": "Guardado \u2713",
    "CARD_TEMP": "Temporal",
    "CARD_SAVED_PARTIAL": "{s}/{t} Guardados",
    "BTN_DETAILS_COLLAPSED": "Detalles \u25bc",
    "BTN_DETAILS_EXPANDED": "Detalles \u25b2",
    "NO_METADATA": "Metadatos del modelo no disponibles (offline o sin metadata.json).",
    "DELETE_REGION": "Eliminar Región",
    "DELETE_SELECTED": "Eliminar Seleccionados",
    "DELETE_ALL": "Eliminar Todos",
    "DELETE_JOB": "Eliminar Todo",
    "VIEW_TILES": "Ver Tiles",
    "HIDE_TILES": "Ocultar",
    "REFRESH_MAP": "Actualizar Mapa",
    "TASK_NAME_PLACEHOLDER": "Ej: Clasificar Amazonia Baja 2025 (Lucas)",
    # M5 - Section titles
    "SECTION_MODEL": "1. Seleccione Modelo (único):",
    "SECTION_REGIONS": "2. Seleccione Regiones:",
    "SECTION_PERIODS": "3. Seleccione Periodos (Año / Año_Mes):",
    "LOADING_CAMPAIGNS": "Cargando campañas...",
    "CAMPAIGN_LABEL": "Campaña:",
    "FILTER_LABEL": "Filtrar:",
    "FILTER_PLACEHOLDER": "buscar por modelo, región, periodo...",
    # M5 - Region diagnostic
    "ERR_CONFIG_ASSET_REGIONS": "CONFIG[asset_regions] no definido",
    "ERR_ASSET_NOT_FOUND": "Asset NO ENCONTRADO en GEE:",
    "ERR_ASSET_EMPTY": "Asset existe pero está VACÍO:",
    "ERR_ASSET_COLUMN": "Columna <code>{prop}</code> no encontrada",
    "ERR_GEE_ACCESS": "Error al acceder a GEE:",
        "WARN_REGION_FALLBACK": "No se pudo poblar regiones automáticamente",
        "ERR_RENDER_PENDING": "Error al renderizar pendientes:",
    "COLUMN_USED": "Columna usada:",
    # M5 - Validation
    "ERR_NO_TASK_NAME": "Atención: Debe asignar un Nombre de Tarea obligatoriamente.",
    "ERR_NO_SELECTION": "Atención: Seleccione 1 Modelo y al menos una Región y un Periodo.",
    # M5 - Feedback messages
    "SAVING_TO_GCS": "Guardando en GCS...",
    "MSG_NO_TASKS_ADDED": "Ninguna tarea fue añadida.",
    "MSG_TASKS_ADDED": "Éxito: {n} tarea(s) agregada(s).",
    "FAIL_SAVE_LOCAL": "fallo al guardar archivo local (bloqueo ocupado)",
    "FAIL_GCS_UPLOAD": "{n} fallaron en GCS",
    "MSG_GCS_OK": "{n} guardadas en GCS.",
    "MSG_SKIPPED_QUEUED": "{n} omitidas (ya en la cola).",
    "WARN_ALREADY_QUEUED": "Atención: {n} tareas ya estaban en la cola.",
    "MSG_LOADED": "{loaded} cargadas, {skipped} omitidas.",
    "TAREA_SAVED": "Tarea guardada en GCS para {m}.",
    "TAREA_DELETED": "Tarea eliminada del GCS: {m}.",
    "WORKPLAN_DISCARDED": "Plano descartado: {m} ({n} jobs removidos).",
    "QUEUE_CLEARED": "Cola vaciada (GCS removido).",
    "SAVED_TO_GCS": "{n} job(s) salvos en GCS.",
    "ALREADY_SAVED": "{n} job(s) ya estaban salvos.",
    "TEMP_JOBS_REMOVED": "{n} job(s) temporários removidos.",
    "NO_TEMP_JOBS": "Ningún job temporário para remover.",
    "REMOVED_FROM_GCS": "{n} job(s) removidos del GCS.",
    "NO_TILES_GCS_ALT": "Ningún tile en GCS.",
    "TILES_DELETED": "{n} tile(s) eliminados.",
    "SELECT_TILES_PROMPT": "Seleccione tiles para eliminar.",
    "SYNC_FROM_GCS": "{n} job(s) sincronizados del GCS.",
    # M5 - Tareas section
    "TAREAS_HEADER": "Tareas en GCS ({n} disponibles)",
    "TAREA_SUMMARY": "Modelo: {m} | Regiones: {r} | Periodos: {p}",
    # M5 - Scale bar
    "SCALE_BAR_WIDTH": "~{w} km de ancho",
    # M5 - Pending debug
    "INFO_FILTER_COUNT": "{total} pendientes | {filtered} tras filtro",
    "GRID_COUNT": "({n} celdas)",
    # M5 - Map legend
    "MAP_LEGEND_COUNTRY": "País",
    "MAP_LEGEND_REGIONS": "Regiones",
    "MAP_LEGEND_GRID": "Grid cim-world",
    # M5 - Model inference errors
    "ERR_MODEL_NO_METADATA": "metadata.json no encontrado en {path}",
    "ERR_MODEL_NO_BANDS_CONFIG": "El modelo no tiene 'bands_config' en los metadatos.",
    "ERR_MODEL_NO_BAND_ORDER": "El modelo no tiene 'band_order' en los metadatos. Reentrena con el M4 actual (commit 9b95392+).",
    "ERR_MODEL_INCONSISTENCY": "Inconsistencia en los metadatos del modelo: {details}",
    # M5 - Jobs details panel
    "JOBS_REGISTERED": "Trabajos registrados",
    "JOBS_SUMMARY": "{total} job(s), {regions} región(es), {periods} periodo(s)",
    "JOB_REGION": "Región",
    "JOB_PERIOD": "Periodo",
    "JOB_STATUS": "Estado",
    "JOB_TASK": "Tarea",
    # M5 - Model metadata panel
    "META_SENSORS": "Sensores",
    "META_PERIODICITY": "Periodicidad",
    "META_BANDS_INPUT": "Bandas (input)",
    "META_ITERATIONS": "Iteraciones",
    "META_LEARNING_RATE": "Learning rate",
    "META_CAMPAIGN": "Campaña",
    # M5 Tabs
    "TAB_GUIDE": "Guía",
    "TAB_REGISTER": "Registrar",
    "TAB_PENDING": "Pendientes",
    "TAB_PUBLISH": "Para Publicar",
    "TAB_MAP": "Mapa",
    "TAB_DONE": "Finalizadas",
    "GUIDE_M5_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:5px;'>M5 - Clasificación Regional de Gran Escala</h3>
        <p>Clasifica múltiples regiones (cartas cim-world-1-250000) usando modelos del M4.</p>
        <h4>Flujo:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_register}</b> — seleccione modelo + regiones + períodos.</li>
            <li><b>{tab_pending}</b> — siga la clasificación tile a tile.</li>
            <li><b>{tab_map}</b> — visión general del procesamiento en vivo.</li>
            <li>Ejecute <code>run_m5_workplan()</code> en el notebook para procesar.</li>
        </ol>
        <h4>Después de la Clasificación (M6):</h4>
        <p>Use <b>M6</b> para mosaicar tiles, generar estadísticas regionales y subir a GEE.
        Abra la UI de M6 y ejecute <code>run_m6_publish()</code> en una celda separada.</p>
        <p>Los trabajos clasificados (COMPLETED) aparecen automáticamente en M6.</p>
    </div>""",
    "GUIDE_M1_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <div style='margin-bottom:15px;padding:10px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px;font-size:13px;'>
            <b>Nota:</b> Actualmente solo los compuestos mensuales para <b>Sentinel-2</b> con <b>minnbr</b> y <b>minnbr_buffer</b> están liberados. Otras combinaciones de sensor/período/mosaico pueden ser experimentales.
        </div>
        <h3 style='color:#2c3e50; border-bottom:2px solid #e67e22; padding-bottom:5px;'>Modulo 1 (M1) — Exportación GEE Multi-Sensor</h3>
        <p>M1 exporta mosaicos satelitales sin nubes desde Google Earth Engine hacia Google Cloud Storage (GCS) o Assets GEE.
        Soporta LANDSAT 5/7, LANDSAT 8/9, Sentinel-2, HLS y MODIS con correcciones radiométricas y máscaras de nubes específicas por sensor.</p>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:15px;'>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Sensores y Fuentes</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>LANDSAT 5/7</b> — archivo histórico (1984–)</li>
                    <li><b>LANDSAT 8/9</b> — OLI/TIRS actual con bandas termales</li>
                    <li><b>Sentinel-2</b> — 10m de resolución, 5 días de revista</li>
                    <li><b>HLS</b> — producto armonizado LANDSAT/Sentinel</li>
                    <li><b>MODIS</b> — cobertura global diaria (250m–1km)</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Metodos de Mosaico</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>minnbr</b> — menor nube según ranking NBR</li>
                    <li><b>minndvi</b> — menor nube según ranking NDVI</li>
                    <li><b>median</b> — composición mediana píxel a pixel</li>
                    <li><b>minnbr_buffer</b> — minnbr con máscara de buffer de fuego INPE</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Flujo de Trabajo</h4>
                <ol style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Seleccione la pestaña del sensor (LANDSAT, S2, HLS, MODIS)</li>
                    <li>Elija el período (mensual / anual)</li>
                    <li>Escoja el método de mosaico</li>
                    <li>Marque las celdas de fecha y banda</li>
                    <li>Haga clic en <b>Iniciar Exportación</b></li>
                </ol>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Información Técnica</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Corrección radiométrica por sensor</li>
                    <li>Mascara de nubes via QA_PIXEL, Fmask, CS+</li>
                    <li>Mascara de buffer de incendios INPE</li>
                    <li>Salida: fragmentos GeoTIFF en GCS / ImageCollection en GEE</li>
                </ul>
            </div>
        </div>
        <div style='margin-top:15px;padding:10px;background:#fef3e2;border-left:4px solid #e67e22;border-radius:4px;font-size:13px;'>
            <b>Consejo:</b> Use <b>Sincronizar Datos</b> para actualizar el cache. Use <b>Seleccionar Pendientes</b> para marcar automáticamente todas las fechas disponibles.
        </div>
    </div>""",
    "GUIDE_M2_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>Modulo 2 (M2) — Montaje de Mosaicos COG</h3>
        <p>M2 ensambla los fragmentos GeoTIFF exportados por M1 en COGs (Cloud-Optimized GeoTIFF) nacionales completos usando GDAL.</p>
        <h4>Flujo:</h4>
        <ol style='line-height:1.6;'>
            <li><b>Verifique el estado</b> — OK (COG existe), READY (fragmentos disponibles), MISS (nada disponible)</li>
            <li><b>Seleccione celdas</b> — marque las bandas que desea ensamblar</li>
            <li>Haga clic en <b>Iniciar Montaje</b></li>
            <li>GDAL descarga fragmentos → construye VRT → convierte a COG (LZW) → sube a GCS</li>
        </ol>
        <h4>Requisitos:</h4>
        <ul style='line-height:1.6;'>
            <li>GDAL instalado (<code>gdalbuildvrt</code>, <code>gdal_translate</code>)</li>
            <li>M1 debe haberse ejecutado primero para producir los fragmentos fuente</li>
        </ul>
        <div style='margin-top:15px;padding:10px;background:#e8f8ed;border-left:4px solid #27ae60;border-radius:4px;font-size:13px;'>
            <b>Consejo:</b> Los COGs se almacenan en <code>.../COG/</code> y son consumidos por M4 (entrenamiento) y M5 (clasificación).
        </div>
    </div>""",
    # Map / Grid
    "LIVE_PROCESSING": "Procesando en vivo",
    "CURRENT_TILE": "Tile actual",
    "COMPLETED": "Completados",
    "GRID_REGION": "Región",
    "GRID_CELLS": "Celdas cim-world",
    # Tiles / GCS
    "TILES": "tiles",
    "MOSAIC": "mosaico",
    "STATS": "stats",
    "SELECT_TILES_DELETE": "Seleccione tiles para eliminar.",
    "TILES_REMOVED": "tiles eliminados.",
    # Task status
    "STATUS_QUEUED": "En el plan",
    "STATUS_RUNNING": "Ejecutando",
    "STATUS_COMPLETED": "Completado",
    "STATUS_FINISHED": "Finalizado",
    "STATUS_ERROR": "Error",
    "STATUS_PUBLISHING": "Publicando",
    "STATUS_PUBLISHED": "Publicado",
    "STATUS_SKIPPED": "Omitido",
    # M4 - Training
        "MODEL_TRAINER": "Entrenador del Modelo",
        "M4_HEADER_TITLE": "M4 - Entrenador del Modelo",
        "SELECTED_OK": "[OK] Seleccionados",
        "TRAINING_PROGRESS": "Entrenando:",
    "ITERATIONS": "Iteraciones",
    "BATCH_SIZE": "Tamaño de Lote",
    "LEARNING_RATE": "Tasa de Aprendizaje",
    "HIDDEN_LAYERS": "Capas Ocultas",
    "ACTIVATION": "Activación",
    "DROPOUT": "Dropout",
    "OPTIMIZER": "Optimizador",
    "LOSS_FN": "Función de Pérdida",
    "METRICS": "Métricas",
    "SAMPLE_SELECTION": "Selección de Muestras",
    "EXTRACTION_MATRIX": "Matriz de Extracción",
    "MODEL_CONFIG": "Configuración del Modelo",
    "GCS_DEST": "Destino GCS",
    "USAGE_GUIDE": "Guía de Uso",
    "GUIDE_M4_HTML": """<div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; background: #fdfdfd; color: #2c3e50; line-height: 1.6;">
        <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">M4 Model Trainer - Guía de Uso</h1>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 20px;">
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #3498db; margin-top:0;">Estructura de la Plataforma</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>{usage_guide}:</b> Pantalla de orientación y documentación.</li>
                    <li><b>{new_training}:</b> Configuración de nuevos experimentos, selección de muestras y bandas.</li>
                    <li><b>{trainings}:</b> Ranking histórico con métricas detalladas y gestión de modelos.</li>
                    <li><b>Canvas:</b> Mesa de auditoria paralela para comparar múltiples modelos en profundidad.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #9b59b6; margin-top:0;">Conceptos Técnicos</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>TensorFlow:</b> Motor de IA de Google para cálculos matemáticos masivos.</li>
                    <li><b>DNN (Deep Neural Network):</b> Red profunda que imita el aprendizaje humano.</li>
                    <li><b>Neuronas:</b> Unidades que procesan señales y activan patrones de aprendizaje.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #e67e22; margin-top:0;">Hiperparametros (DNN)</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Layers:</b> Arquitectura de la red. Más capas captan detalles más finos.</li>
                    <li><b>Learning Rate (LR):</b> Controla que tan rápido se ajusta el modelo.</li>
                    <li><b>Epochs:</b> Ciclos de entrenamiento completos sobre el set de muestras.</li>
                    <li><b>Batch Size:</b> Bloques de datos procesados antes de cada actualizacion.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #27ae60; margin-top:0;">Diccionario de Calidad</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Accuracy:</b> Porcentaje total de aciertos globales.</li>
                    <li><b>Precision:</b> Fidelidad: ¿Cuánto del fuego marcado es real? (Evita falsos positivos).</li>
                    <li><b>Recall:</b> Cobertura: ¿Cuánto del fuego real se encontró? (Evita omisiones).</li>
                    <li><b>F1-Score:</b> Media armónica. El mejor balance entre Precision y Recall.</li>
                    <li><b>Nota IA:</b> Auditoría automática que castiga severamente las omisiones.</li>
                    <li><b>Nota Humana:</b> Evaluación subjetiva (1-5) sobre el Espacio Latente.</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 30px; padding: 15px; background: #e8f4fd; border-left: 5px solid #3498db; border-radius: 4px; font-size:14px;">
            <b>[Consejo] Pro-Tip del Auditor:</b> Use el <b>Canvas</b> para cargar un modelo antiguo (benchmark) y su modelo nuevo. Compare si la separación de clases en t-SNE 3D ha mejorado o si hay nuevas zonas de confusión.
        </div>
    </div>""",
    # M4 - Canvas
    "METADATA": "Metadatos",
    "KPIS": "KPIs",
    "CONFUSION": "Confusión",
    "HISTORY": "Historial",
    "PROB": "Prob",
    "PR_CURVE": "PR-Curve",
    "MANAGEMENT": "Gestión",
    "RANKING": "Ranking / Repositorio",
    "SELECTED_CANVAS": "Seleccionados en Canvas",
    "SEARCH_REPO": "Buscar en repositorio...",
    "SEARCH_SAMPLES": "Buscar muestras...",
    "SYNC_CATALOG": "Sincronizar Catálogo (GCS)",
    "APPLY_VISIBILITY": "Aplicar Visibilidad",
    # M1 / M2
    "SYNC_DATA": "Sincronizar Datos",
    "SELECT_PENDING": "Seleccionar Pendientes",
    "CLEAR_SELECTION": "Limpiar Seleccion",
    "EXPORT_START": "Iniciar Exportación",
    "MOSAIC_START": "Iniciar Montaje",
    # M6
    "POST_CLASSIFICATION": "Procesador Post-Clasificación",
    "USING_PRESET": "Usando Configuración Preset",
    "FILTER_START": "Iniciar Filtrado",
    "EXPORT_TASK_STARTED": "Tarea de exportación iniciada",
    "CONFIG_SUMMARY": "Resumen de Configuración Usada",
    # M7
    "CURATOR": "Curador",
    "CURATOR_DESC": "Publicación de Colección Pre-Oficial",
    "USING_VOTES_PRESET": "Usando Votacion Preset",
    "CURATION_START": "Iniciar Curaduría",
    "EXPORT_ASSET_STARTED": "Exportación GEE Asset iniciada",
    # Cache
    "CACHE_REMOVED": "Caché local removido",
    "CACHE_NOT_FOUND": "Caché no encontrado",
    # Misc
    "LOADING_TILES": "Cargando tiles...",
    "CONFIRM_DELETE": "Confirmar Eliminacion",
    "CONFIRM_DELETE_ALL": "Confirmar Eliminacion Total",
    "CANCELED": "Cancelado.",
    "GLOBAL_OPTS_SET": "Opciones globales configuradas",
    # Dropdown labels
    "DROP_MODEL": "Modelo:",
    "DROP_REGION": "Región:",
    "DROP_YEAR": "Año:",
    "DROP_TASK": "Tarea:",
    "DROP_TASK_NAME": "Nombre Tarea:",
    # Hyperparams
    "HP_EPOCHS": "Épocas",
    "HP_PATIENCE": "Paciencia",
    "HP_TEST_SPLIT": "Split de Test",
    "HP_BALANCE": "Balanceo",
    "HP_AUGMENT": "Aumento de Datos",
    "HP_ACTIVATION": "Función de Activación",
    "HP_OPTIMIZER": "Optimizador",
    "HP_LOSS": "Función de Pérdida",
    # Extraction
    "EXTRACTION_TITLE": "Matriz de Extracción (Multisensor GCS)",
    "SAMPLE_SELECT": "Seleccion de Muestras",
    "AVAILABLE": "Disponibles",
    "SELECTED": "Seleccionados",
    "ADD": "Agregar",
    "REMOVE": "Remover",
    "CAMPAIGN": "Campaña",
    "SENSOR": "Sensor",
    "BANDS": "Bandas",
    "STATUS_OK": "OK",
    "STATUS_RUN": "Ejec.",
    "STATUS_MISS": "Falt.",
    # Training
    "TRAINING_ID": "ID Training:",
    "SHORTNAME": "Nombre Rapido:",
    "COMMENTS": "Comentarios...",
    "START_TRAINING": "Iniciar Entrenamiento",
    "CANVAS_TITLE": "Centro de Entrenamientos y Auditoría",
    "CANVAS_EMPTY": "Canvas vacío",
    "CANVAS_HINT": "Busque y seleccione modelos en el panel lateral para visualizarlos aquí.",
    "LOSS_ACC": "Loss / Accuracy",
    "VIZ_METADATA": "Metadatos",
    "VIZ_KPIS": "KPIs",
    "VIZ_CONFUSION": "Confusión",
    "VIZ_HISTORY": "Historial",
    "VIZ_PROB": "Prob",
    "VIZ_PR_CURVE": "PR-Curve",
    "VIZ_MANAGEMENT": "Gestion",
    # Repository
    "REPO_TITLE": "Ranking / Repositorio",
    "REPO_SEARCH": "Buscar en repositorio...",
    "REPO_SYNC": "Sincronizar GCS",
    "REPO_ALL": "Todos",
    "REPO_CLEAR": "Limpiar",
    "REPO_SYNC_CATALOG": "Sincronizar Catalogo (GCS)",
    "REPO_SCANNING": "Escaneando GCS... Aguarde.",
    "REPO_SCAN_DONE": "¡Catálogo Sincronizado!",
    # Tab titles
    "SORT_BY": "Ordenar:",
    "NEW_TRAINING": "Nuevo Entrenamiento",
    "TRAININGS": "Entrenamientos",
    # M1 - Export
    "EXPORT_TITLE": "Exportación de Colecciones",
    "EXPORT_SYNC": "Sincronizar Datos",
    "EXPORT_SELECT": "Seleccionar Pendientes",
    "EXPORT_CLEAR": "Limpiar Seleccion",
    "EXPORT_START_BTN": "Iniciar Exportación",
    "EXPORT_SENT": "tareas enviadas.",
    "EXPORT_NONE_SEL": "Ninguna opción seleccionada para exportación.",
    # M2 - Mosaic
    "MOSAIC_TITLE": "Montaje de Mosaicos",
    "MOSAIC_SYNC": "Sincronizar Datos",
    "MOSAIC_SELECT": "Seleccionar Pendientes",
    "MOSAIC_CLEAR": "Limpiar Seleccion",
    "MOSAIC_START_BTN": "Iniciar Montaje",
    "MOSAIC_DONE": "Montaje concluido con éxito",
    "M1_HEADER_TITLE": "M1 - Despachador",
    "M1_HEADER_SUBTITLE": "Interfaz multisensor para exportación",
    "M1_TITLE_FULL": "M1 - Despachador de Mosaicos",
    "M1_DESCRIPTION": "Interfaz multisensor para despachar composiciones (Assets/GCS) a la nube.",
    "LABEL_PROJECT": "Proyecto",
    "BTN_SELECT_ROW": "[S]",
    "COL_TYPE": "Tipo",
    "M2_HEADER_TITLE": "M2 - Montador",
    "M2_HEADER_SUBTITLE": "Interfaz para montaje de mosaicos nacionales (COG)",
    "M2_CONSTRUCTOR_TITLE": "M2 - Montador de Mosaicos (COG)",
    "M2_CONSTRUCTOR_DESC": "Interfaz para convertir chunks GCS en mosaicos COG nacionales.",
    "STATUS_READY": "LISTO",
    # M0 - Startup
    "START_COUNTRY": "País",
    "START_BUCKET": "Bucket",
    "START_VERSION": "Version",
    "START_SENSOR": "Sensor",
    "START_PERIODICITY": "Periodicidad",
    "START_CAMPAIGN": "Campaña",
    "START_FOUND": "encontrado",
    "START_NOT_FOUND": "no encontrado",
    "START_GDAL_OK": "GDAL encontrado y agregado al PATH",
    "START_GDAL_MISSING": "Aviso: Utilitarios GDAL no encontrados",
    "START_COLAB_HINT": "En Google Colab, ejecute: !apt-get install -y gdal-bin",
    "START_WIN_HINT": "En Windows, asegúrese de que GDAL esté en su PATH",
    # M6
    "M6_TITLE": "Procesador Post-Clasificación",
    "M6_PRESET": "Usando Configuración Preset",
    "M6_DOWNLOAD": "Descargando fragmentos",
    "M6_VRT": "Construyendo VRT",
    "M6_COG": "Convirtiendo a COG",
    "M6_UPLOAD": "Subido",
    "M6_START": "Iniciando Filtrado",
    "M6_EXPORT_OK": "Tarea de exportación iniciada",
    "M6_SUMMARY": "Resumen de Configuración Usada",
    "TAB_ANALYTICS": "Analíticas",
    "TAB_M6_COVERAGE": "Cobertura",
    "GUIDE_M6_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>M6 - Mosaico, Stats y Publicación</h3>
        <p>Publica tiles clasificados de M5: crea mosaicos regionales, calcula estadísticas de área quemada y sube a Google Earth Engine.</p>
        <h4>Flujo:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_publish}</b> — grupos de tiles clasificados esperando mosaico.</li>
            <li><b>{tab_done}</b> — regiones publicadas con mosaico y estadísticas.</li>
            <li><b>{tab_analytics}</b> — tabla consolidada de estadísticas con descarga.</li>
            <li><b>{tab_coverage}</b> — mapa de cobertura: publicado vs pendiente.</li>
            <li>Ejecute <code>run_m6_publish()</code> en el notebook para procesar.</li>
        </ol>
    </div>""",
    "DOWNLOAD_TABLE": "Descargar Tabla",
    "ANALYTICS_FILTER_MODEL": "Modelo",
    "ANALYTICS_FILTER_REGION": "Región",
    "ANALYTICS_FILTER_PERIOD": "Periodo",
    "REFRESH_M6": "Actualizar M6",
    "M6_HEADER_TITLE": "M6 - Mosaico, Stats y Publicación",
    "M6_LABEL_PERIOD": "Período:",
    "M6_GROUPS_PENDING": "{n} grupos pendientes de mosaico",
    "M6_MOSAIC_OK": "mosaico OK",
    "M6_DOWNLOAD_MOSAIC": "OK \u2197",
    "M6_PUBLISHED_GROUPS": "{n} grupos publicados",
    "M6_BADGE_MOSAIC": "Mosaico",
    "M6_BADGE_STATS": "Estats",
    "M6_BADGE_GEE": "GEE",
    "M6_COL_PCT": "%",
    "M6_COL_HA": "ha",
    "M6_COL_TILES": "Tiles",
    "M6_LEGEND_PUBLISHED": "Publicado",
    "M6_LEGEND_PARTIAL": "Parcial",
    "M6_LEGEND_CLASSIFIED_ONLY": "Solo clasificado",
    "M6_LEGEND_NO_DATA": "Sin datos",
    "M6_N_RECORDS": "{n} registros",
    "M6_DOWNLOAD_LINK": "Descargar {fname}",
    "M6_NO_STATS": "No hay estadísticas consolidadas. Ejecute M6 publish primero.",
    "M6_NO_MATCHING": "No hay registros coincidentes.",
    "M6_NO_CLASSIFIED_GROUPS": "No se encontraron grupos clasificados.",
    # M7
    "M7_TITLE": "Curador",
    "M7_DESC": "Publicación de Colección Pre-Oficial",
    "M7_PRESET": "Usando Votacion Preset",
    "M7_START": "Iniciando Curaduría",
    "M7_EXPORT_OK": "Exportación GEE Asset iniciada para",
    "M7_SUMMARY": "Resumen de Configuración Usada",
    # M3
    "M3_TITLE": "M3 - Colecta de Muestras (GEE Toolkit Gateway)",
    "M3_SOURCE": "Acceso al codigo fuente (GitHub)",
    "M3_EDITOR": "Acceso directo (Editor GEE)",
    "M3_DOCS": "Documentacion y normas de uso",
    "M3_SLIDES": "Presentacion tutorial (Google Slides)",
    "M3_INTRO_LINE1": "La colecta de muestras se realiza exclusivamente",
    "M3_INTRO_LINE2": "a través de la interfaz JavaScript en Google Earth Engine.",
    # M4 - Analytics / KPIs
    "ACCURACY": "Precisión",
    "PRECISION": "Precisión",
    "RECALL": "Exhaustividad",
    "F1_SCORE": "F1-Score",
    "AUTO_RATING": "Nota Automatica",
    "HUMAN_NOTE": "Nota Humana",
    "NO_METRICS": "Sin métricas",
    "CONFUSION_MATRIX": "Matriz de Confusión (%)",
    "HISTORICAL_EVOLUTION": "Evolución Histórica",
    "PROB_DISTRIBUTION": "Distribución de Probabilidades",
    "CONFIDENCE": "Confianza",
    "LATENT_PROJ_2D": "Proyección Latente 2D",
    "COST_LOSS": "Costo (Loss)",
    "CLASSIC_METRICS": "Métricas Clásicas y Proyecciones Estáticas",
    "INTERACTIVE_LATENT": "Espacio Latente Interactivo",
    "PCA_3D": "PCA 3D",
    "TSNE_3D": "t-SNE 3D",
    "PCA_3D_INTERACTIVE": "PCA 3D Interactivo",
    "TSNE_3D_INTERACTIVE": "t-SNE 3D Interactivo",
    "RETRAIN": "Reentrenar",
    # M4 - Fire class labels
    "FIRE": "Fuego",
    "NO_FIRE": "No-fuego",
    "FIRE_CLASS": "Fuego",
    "NO_FIRE_CLASS": "No-fuego",
    # M4 - UI Labels
    "BASIC_STATS": "Estadísticas Básicas",
    "PCA_LATENT": "Espacio Latente PCA",
    "TSNE_LATENT": "Espacio Latente t-SNE",
    "VIZ_OPTIONS": "Opciones de Visualización",
    "VIZ_PCA2D": "PCA 2D",
    "VIZ_PCA3D_STATIC": "PCA 3D (Estático)",
    "VIZ_PCA3D_INTERACTIVE": "PCA 3D (Interactivo)",
    "VIZ_TSNE3D_STATIC": "t-SNE 3D (Estático)",
    "VIZ_TSNE3D_INTERACTIVE": "t-SNE 3D (Interactivo)",
    "HYPERPARAMS_SECTION": "Hiperparámetros (DNN)",
    "LIVE_TRAINING": "Entrenamiento en Vivo",
    "TRAINING_IN_PROGRESS": "Entrenamiento en curso...",
    "LIVE_TSNE_AUDIT": "Auditoría t-SNE en Vivo (Espacio Latente)",
    "ADD_TO_CANVAS": "Agregar al Canvas",
    "REMOVE_FROM_CANVAS": "Retirar del Canvas",
    "BTN_CLOSE": "X",
    "RELOAD_SAMPLES": "Recargar lista de muestras del GCS",
    "PLUS": "+ ",
    # M4 - Analytics card labels
    "LAYERS_LABEL": "Capas:",
    "LR_ABBR": "LR:",
    "SAMPLES_LABEL": "Muestras:",
    "NO_COMMENTS": "Sin comentarios.",
    "HIDE_ADVANCED": "Ocultar parámetros avanzados",
    "SHOW_ALL_PARAMS": "Mostrar todos los parámetros",
    "TSNE_AXIS_1": "t-SNE 1",
    "TSNE_AXIS_2": "t-SNE 2",
    "TSNE_AXIS_3": "t-SNE 3",
    "ACC_ABBR": "Prec",
    "F1_ABBR": "F1",
}

STRINGS_PT = {
    # General
    "LOADING": "Carregando...",
    "PROCESSING": "Processando...",
    "DELETING": "Excluindo...",
    "SAVING": "Salvando...",
    "SYNCING": "Sincronizando...",
    "SYNCING_TASKS": "Sincronizando tarefas GEE...",
    "SEARCHING": "Buscando...",
    "DONE": "Feito",
    "ERROR": "Erro",
    "SUCCESS": "Sucesso",
    "WARNING": "Atenção",
    "INFO": "Informação",
    "ALL": "Todos",
    "ALL_F": "Todas",
    "NONE": "Nenhum",
    "BACK": "Voltar",
    "OK": "OK",
    "CONFIRM": "Confirmar",
    "CANCEL": "Cancelar",
    "SAVE": "Salvar",
    "DELETE": "Excluir",
    "SYNC": "Sincronizar",
    "SEARCH": "Buscar",
    "CLEAR": "Limpar",
    "REFRESH": "Atualizar",
    "APPLY": "Aplicar",
    "CLOSE": "Fechar",
    "MODEL": "Modelo",
    "REGION": "Região",
    "YEAR": "Ano",
    "PERIOD": "Periodo",
    "TASK": "Tarefa",
    "TASK_NAME": "Nome Tarefa",
    "STATUS": "Estado",
    "PROGRESS": "Progresso",
    "DESCRIPTION": "Descrição",
    "NAME": "Nome",
    "DATE": "Data",
    "ID": "ID",
    # Empty states
    "NO_DATA": "Nenhum dado disponível.",
    "NO_RESULTS": "Nenhum resultado encontrado.",
    "NO_TASKS": "Nenhuma tarefa pendente.",
    "NO_TASKS_PUBLISH": "Nenhuma tarefa pronta para publicar.",
    "NO_TASKS_DONE": "Nenhuma tarefa finalizada ainda.",
    "NO_TILES_GCS": "Nenhum tile no GCS.",
    "NO_MAP": "Não foi possível gerar o mapa. Verifique conexão GEE.",
    "NO_SELECTION": "Nenhuma opção selecionada.",
    "NO_SAMPLES": "Nenhuma amostra encontrada com este filtro.",
    "NO_COGS": "Nenhum COG encontrado no repositório GCS.",
    # M5 Widgets
    "ADD_BATCH": "Adicionar Lote ao Plano",
    "REFRESH_VIEW": "Atualizar Vista",
    "LOAD_TO_QUEUE": "Carregar no Plano",
    "CLEAR_TEMP_TASKS": "Limpar Tarefas Temporárias",
    "SAVE_TASK_GCS": "Salvar Tarefa GCS",
    "EXCLUDE_TASK_GCS": "Excluir Tarefa GCS",
    "DELETE_MODEL": "Excluir Modelo",
    "DISCARD_WORKPLAN": "Descartar Plano de Trabalho",
    "GLOBAL_ACTIONS": "Ações Globais",
    "NO_PENDING_JOBS": "Nenhum trabalho pendente para classificar.",
        "ERR_NO_SAMPLES": "Erro: Nenhuma amostra selecionada.",
        "ERR_NO_BANDS": "Erro: Nenhuma banda selecionada na Matriz de Extração.",
        "ERR_INCOMPATIBLE_ENV": "Ambiente local incompatível",
        "ERR_NO_AVX": "Sua CPU não suporta as instruções AVX/AVX2 requeridas pelo TensorFlow.",
        "ERR_RUN_ON_COLAB": "Por favor: Execute este treinamento no Google Colab.",
    "CARD_SAVED": "Salvo \u2713",
    "CARD_TEMP": "Temporário",
    "CARD_SAVED_PARTIAL": "{s}/{t} Salvos",
    "BTN_DETAILS_COLLAPSED": "Detalhes \u25bc",
    "BTN_DETAILS_EXPANDED": "Detalhes \u25b2",
    "NO_METADATA": "Metadados do modelo indisponíveis (offline ou sem metadata.json).",
    "DELETE_REGION": "Excluir Região",
    "DELETE_SELECTED": "Excluir Selecionados",
    "DELETE_ALL": "Excluir Todos",
    "DELETE_JOB": "Excluir Tudo",
    "VIEW_TILES": "Ver Tiles",
    "HIDE_TILES": "Ocultar",
    "REFRESH_MAP": "Atualizar Mapa",
    "TASK_NAME_PLACEHOLDER": "Ex: Classificar Amazônia Baixa 2025 (Lucas)",
    # M5 - Section titles
    "SECTION_MODEL": "1. Selecione Modelo (único):",
    "SECTION_REGIONS": "2. Selecione Regiões:",
    "SECTION_PERIODS": "3. Selecione Períodos (Ano / Ano_Mês):",
    "LOADING_CAMPAIGNS": "Carregando campanhas...",
    "CAMPAIGN_LABEL": "Campanha:",
    "FILTER_LABEL": "Filtrar:",
    "FILTER_PLACEHOLDER": "buscar por modelo, região, período...",
    # M5 - Region diagnostic
    "ERR_CONFIG_ASSET_REGIONS": "CONFIG[asset_regions] não definido",
    "ERR_ASSET_NOT_FOUND": "Asset NÃO ENCONTRADO no GEE:",
    "ERR_ASSET_EMPTY": "Asset existe mas está VAZIO:",
    "ERR_ASSET_COLUMN": "Coluna <code>{prop}</code> não encontrada",
    "ERR_GEE_ACCESS": "Erro ao acessar GEE:",
        "WARN_REGION_FALLBACK": "Não foi possível popular regiões automaticamente",
        "ERR_RENDER_PENDING": "Erro ao renderizar pendentes:",
    "COLUMN_USED": "Coluna usada:",
    # M5 - Validation
    "ERR_NO_TASK_NAME": "Atenção: Deve atribuir um Nome de Tarefa obrigatoriamente.",
    "ERR_NO_SELECTION": "Atenção: Selecione 1 Modelo e pelo menos uma Região e um Período.",
    # M5 - Feedback messages
    "SAVING_TO_GCS": "Salvando no GCS...",
    "MSG_NO_TASKS_ADDED": "Nenhuma tarefa foi adicionada.",
    "MSG_TASKS_ADDED": "Sucesso: {n} tarefa(s) adicionada(s).",
    "FAIL_SAVE_LOCAL": "falha ao salvar arquivo local (lock ocupado)",
    "FAIL_GCS_UPLOAD": "{n} falharam no GCS",
    "MSG_GCS_OK": "{n} salvas no GCS.",
    "MSG_SKIPPED_QUEUED": "{n} omitidas (já na fila).",
    "WARN_ALREADY_QUEUED": "Atenção: {n} tarefas já estavam na fila.",
    "MSG_LOADED": "{loaded} carregadas, {skipped} omitidas.",
    "TAREA_SAVED": "Tarefa salva no GCS para {m}.",
    "TAREA_DELETED": "Tarefa excluída do GCS: {m}.",
    "WORKPLAN_DISCARDED": "Plano descartado: {m} ({n} jobs removidos).",
    "QUEUE_CLEARED": "Fila esvaziada (GCS removido).",
    "SAVED_TO_GCS": "{n} job(s) salvos no GCS.",
    "ALREADY_SAVED": "{n} job(s) já estavam salvos.",
    "TEMP_JOBS_REMOVED": "{n} job(s) temporários removidos.",
    "NO_TEMP_JOBS": "Nenhum job temporário para remover.",
    "REMOVED_FROM_GCS": "{n} job(s) removidos do GCS.",
    "NO_TILES_GCS_ALT": "Nenhum tile no GCS.",
    "TILES_DELETED": "{n} tile(s) eliminados.",
    "SELECT_TILES_PROMPT": "Selecione tiles para eliminar.",
    "SYNC_FROM_GCS": "{n} job(s) sincronizados do GCS.",
    # M5 - Tareas section
    "TAREAS_HEADER": "Tarefas no GCS ({n} disponíveis)",
    "TAREA_SUMMARY": "Modelo: {m} | Regiões: {r} | Períodos: {p}",
    # M5 - Scale bar
    "SCALE_BAR_WIDTH": "~{w} km de largura",
    # M5 - Pending debug
    "INFO_FILTER_COUNT": "{total} pendentes | {filtered} após filtro",
    "GRID_COUNT": "({n} células)",
    # M5 - Map legend
    "MAP_LEGEND_COUNTRY": "País",
    "MAP_LEGEND_REGIONS": "Regiões",
    "MAP_LEGEND_GRID": "Grid cim-world",
    # M5 - Model inference errors
    "ERR_MODEL_NO_METADATA": "metadata.json não encontrado em {path}",
    "ERR_MODEL_NO_BANDS_CONFIG": "Modelo não tem 'bands_config' nos metadados.",
    "ERR_MODEL_NO_BAND_ORDER": "Modelo não tem 'band_order' nos metadados. Retreine com o M4 atual (commit 9b95392+).",
    "ERR_MODEL_INCONSISTENCY": "Inconsistência nos metadados do modelo: {details}",
    # M5 - Jobs details panel
    "JOBS_REGISTERED": "Trabalhos registrados",
    "JOBS_SUMMARY": "{total} job(s), {regions} região(ões), {periods} período(s)",
    "JOB_REGION": "Região",
    "JOB_PERIOD": "Período",
    "JOB_STATUS": "Status",
    "JOB_TASK": "Tarefa",
    # M5 - Model metadata panel
    "META_SENSORS": "Sensores",
    "META_PERIODICITY": "Periodicidade",
    "META_BANDS_INPUT": "Bandas (input)",
    "META_ITERATIONS": "Iterações",
    "META_LEARNING_RATE": "Learning rate",
    "META_CAMPAIGN": "Campanha",
    # M5 Tabs
    "TAB_GUIDE": "Guia",
    "TAB_REGISTER": "Registrar",
    "TAB_PENDING": "Pendentes",
    "TAB_PUBLISH": "Para Publicar",
    "TAB_MAP": "Mapa",
    "TAB_DONE": "Finalizadas",
    "GUIDE_M5_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:5px;'>M5 - Classificação Regional de Grande Escala</h3>
        <p>Classifica múltiplas regiões (cartas cim-world-1-250000) usando modelos do M4.</p>
        <h4>Fluxo:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_register}</b> — selecione modelo + regiões + períodos.</li>
            <li><b>{tab_pending}</b> — acompanhe a classificação tile a tile.</li>
            <li><b>{tab_map}</b> — visão geral do processamento ao vivo.</li>
            <li>Execute <code>run_m5_workplan()</code> no notebook para processar.</li>
        </ol>
        <h4>Após a Classificação (M6):</h4>
        <p>Use o <b>M6</b> para mosaicar tiles, gerar estatísticas regionais e enviar para o GEE.
        Abra a UI do M6 e execute <code>run_m6_publish()</code> em uma célula separada.</p>
        <p>Os trabalhos classificados (COMPLETED) aparecem automaticamente no M6.</p>
    </div>""",
    "GUIDE_M1_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <div style='margin-bottom:15px;padding:10px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px;font-size:13px;'>
            <b>Nota:</b> Atualmente apenas os compostos mensais para <b>Sentinel-2</b> com <b>minnbr</b> e <b>minnbr_buffer</b> estão liberados. Outras combinações de sensor/período/mosaico podem ser experimentais.
        </div>
        <h3 style='color:#2c3e50; border-bottom:2px solid #e67e22; padding-bottom:5px;'>Modulo 1 (M1) — Exportação GEE Multi-Sensor</h3>
        <p>M1 exporta mosaicos de satelite sem nuvens do Google Earth Engine para o Google Cloud Storage (GCS) ou Assets GEE.
        Suporta LANDSAT 5/7, LANDSAT 8/9, Sentinel-2, HLS e MODIS com correções radiométricas e máscaras de nuvens específicas por sensor.</p>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:15px;'>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Sensores e Fontes</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>LANDSAT 5/7</b> — arquivo histórico (1984–)</li>
                    <li><b>LANDSAT 8/9</b> — OLI/TIRS atual com bandas termais</li>
                    <li><b>Sentinel-2</b> — 10m de resolução, 5 dias de revisita</li>
                    <li><b>HLS</b> — produto harmonizado LANDSAT/Sentinel</li>
                    <li><b>MODIS</b> — cobertura global diária (250m–1km)</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Metodos de Mosaico</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>minnbr</b> — menor nuvem segundo ranking NBR</li>
                    <li><b>minndvi</b> — menor nuvem segundo ranking NDVI</li>
                    <li><b>median</b> — composição mediana pixel a pixel</li>
                    <li><b>minnbr_buffer</b> — minnbr com máscara de buffer de fogo INPE</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Fluxo de Trabalho</h4>
                <ol style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Selecione a aba do sensor (LANDSAT, S2, HLS, MODIS)</li>
                    <li>Escolha o período (mensal / anual)</li>
                    <li>Selecione o método de mosaico</li>
                    <li>Marque as células de data e banda</li>
                    <li>Clique em <b>Iniciar Exportação</b></li>
                </ol>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Informação Técnica</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Correção radiométrica por sensor</li>
                    <li>Mascara de nuvens via QA_PIXEL, Fmask, CS+</li>
                    <li>Mascara de buffer de incendios INPE</li>
                    <li>Saída: fragmentos GeoTIFF no GCS / ImageCollection no GEE</li>
                </ul>
            </div>
        </div>
        <div style='margin-top:15px;padding:10px;background:#fef3e2;border-left:4px solid #e67e22;border-radius:4px;font-size:13px;'>
            <b>Dica:</b> Use <b>Sincronizar Dados</b> para atualizar o cache. Use <b>Selecionar Pendentes</b> para marcar automáticamente todas as datas disponíveis.
        </div>
    </div>""",
    "GUIDE_M2_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>Modulo 2 (M2) — Montagem de Mosaicos COG</h3>
        <p>M2 monta os fragmentos GeoTIFF exportados pelo M1 em COGs (Cloud-Optimized GeoTIFF) nacionais completos usando GDAL.</p>
        <h4>Fluxo:</h4>
        <ol style='line-height:1.6;'>
            <li><b>Verifique o status</b> — OK (COG existe), READY (fragmentos disponíveis), MISS (nada disponivel)</li>
            <li><b>Selecione celulas</b> — marque as bandas que deseja montar</li>
            <li>Clique em <b>Iniciar Montagem</b></li>
            <li>GDAL baixa fragmentos → constrói VRT → converte para COG (LZW) → envia ao GCS</li>
        </ol>
        <h4>Requisitos:</h4>
        <ul style='line-height:1.6;'>
            <li>GDAL instalado (<code>gdalbuildvrt</code>, <code>gdal_translate</code>)</li>
            <li>M1 deve ter sido executado primeiro para produzir os fragmentos fonte</li>
        </ul>
        <div style='margin-top:15px;padding:10px;background:#e8f8ed;border-left:4px solid #27ae60;border-radius:4px;font-size:13px;'>
            <b>Dica:</b> Os COGs são armazenados em <code>.../COG/</code> e consumidos pelo M4 (treinamento) e M5 (classificação).
        </div>
    </div>""",
    # Map / Grid
    "LIVE_PROCESSING": "Processando ao vivo",
    "CURRENT_TILE": "Tile atual",
    "COMPLETED": "Completados",
    "GRID_REGION": "Região",
    "GRID_CELLS": "Células cim-world",
    # Tiles / GCS
    "TILES": "tiles",
    "MOSAIC": "mosaico",
    "STATS": "stats",
    "SELECT_TILES_DELETE": "Selecione tiles para excluir.",
    "TILES_REMOVED": "tiles excluídos.",
    # Task status
    "STATUS_QUEUED": "No plano",
    "STATUS_RUNNING": "Executando",
    "STATUS_COMPLETED": "Completado",
    "STATUS_FINISHED": "Finalizado",
    "STATUS_ERROR": "Erro",
    "STATUS_PUBLISHING": "Publicando",
    "STATUS_PUBLISHED": "Publicado",
    "STATUS_SKIPPED": "Omitido",
    # M4 - Training
        "MODEL_TRAINER": "Treinador de Modelo",
        "M4_HEADER_TITLE": "M4 - Treinador de Modelo",
        "SELECTED_OK": "[OK] Selecionados",
        "TRAINING_PROGRESS": "Treinando:",
    "ITERATIONS": "Iterações",
    "BATCH_SIZE": "Tamanho do Lote",
    "LEARNING_RATE": "Taxa de Aprendizado",
    "HIDDEN_LAYERS": "Camadas Ocultas",
    "ACTIVATION": "Ativação",
    "DROPOUT": "Dropout",
    "OPTIMIZER": "Otimizador",
    "LOSS_FN": "Função de Perda",
    "METRICS": "Métricas",
    "SAMPLE_SELECTION": "Seleção de Amostras",
    "EXTRACTION_MATRIX": "Matriz de Extração",
    "MODEL_CONFIG": "Configuração do Modelo",
    "GCS_DEST": "Destino GCS",
    "USAGE_GUIDE": "Guia de Uso",
    "GUIDE_M4_HTML": """<div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; background: #fdfdfd; color: #2c3e50; line-height: 1.6;">
        <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">M4 Model Trainer - Guia de Uso</h1>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 20px;">
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #3498db; margin-top:0;">Estrutura da Plataforma</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>{usage_guide}:</b> Tela de orientação e documentação.</li>
                    <li><b>{new_training}:</b> Configuração de novos experimentos, seleção de amostras e bandas.</li>
                    <li><b>{trainings}:</b> Ranking histórico com métricas detalhadas e gestão de modelos.</li>
                    <li><b>Canvas:</b> Mesa de auditoria paralela para comparar múltiplos modelos em profundidade.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #9b59b6; margin-top:0;">Conceitos Técnicos</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>TensorFlow:</b> Motor de IA do Google para cálculos matemáticos massivos.</li>
                    <li><b>DNN (Deep Neural Network):</b> Rede profunda que imita o aprendizado humano.</li>
                    <li><b>Neurônios:</b> Unidades que processam sinais e ativam padrões de aprendizado.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #e67e22; margin-top:0;">Hiperparametros (DNN)</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Layers:</b> Arquitetura da rede. Mais camadas captam detalhes mais finos.</li>
                    <li><b>Learning Rate (LR):</b> Controla a velocidade de ajuste do modelo.</li>
                    <li><b>Epochs:</b> Ciclos completos de treinamento sobre o conjunto de amostras.</li>
                    <li><b>Batch Size:</b> Blocos de dados processados antes de cada atualização.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #27ae60; margin-top:0;">Dicionário de Qualidade</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Accuracy:</b> Porcentagem total de acertos globais.</li>
                    <li><b>Precision:</b> Fidelidade: Quanto do fogo marcado é real? (Evita falsos positivos).</li>
                    <li><b>Recall:</b> Cobertura: Quanto do fogo real foi encontrado? (Evita omissões).</li>
                    <li><b>F1-Score:</b> Média harmônica. O melhor equilibrio entre Precision e Recall.</li>
                    <li><b>Nota IA:</b> Auditoria automática que pune severamente as omissões.</li>
                    <li><b>Nota Humana:</b> Avaliação subjetiva (1-5) sobre o Espaco Latente.</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 30px; padding: 15px; background: #e8f4fd; border-left: 5px solid #3498db; border-radius: 4px; font-size:14px;">
            <b>[Dica] Pro-Tip do Auditor:</b> Use o <b>Canvas</b> para carregar um modelo antigo (benchmark) e seu modelo novo. Compare se a separação de classes em t-SNE 3D melhorou ou se há novas zonas de confusão.
        </div>
    </div>""",
    # M4 - Canvas
    "METADATA": "Metadados",
    "KPIS": "KPIs",
    "CONFUSION": "Confusão",
    "HISTORY": "Historico",
    "PROB": "Prob",
    "PR_CURVE": "PR-Curve",
    "MANAGEMENT": "Gestão",
    "RANKING": "Ranking / Repositorio",
    "SELECTED_CANVAS": "Selecionados no Canvas",
    "SEARCH_REPO": "Buscar no repositorio...",
    "SEARCH_SAMPLES": "Buscar amostras...",
    "SYNC_CATALOG": "Sincronizar Catálogo (GCS)",
    "APPLY_VISIBILITY": "Aplicar Visibilidade",
    # M1 / M2
    "SYNC_DATA": "Sincronizar Dados",
    "SELECT_PENDING": "Selecionar Pendentes",
    "CLEAR_SELECTION": "Limpar Seleção",
    "EXPORT_START": "Iniciar Exportação",
    "MOSAIC_START": "Iniciar Montagem",
    # M6
    "POST_CLASSIFICATION": "Processador Pós-Classificação",
    "USING_PRESET": "Usando Configuração Predefinida",
    "FILTER_START": "Iniciar Filtragem",
    "EXPORT_TASK_STARTED": "Tarefa de exportação iniciada",
    "CONFIG_SUMMARY": "Resumo da Configuração",
    # M7
    "CURATOR": "Curador",
    "CURATOR_DESC": "Publicação de Coleção Pré-Oficial",
    "USING_VOTES_PRESET": "Usando Votação Predefinida",
    "CURATION_START": "Iniciar Curadoria",
    "EXPORT_ASSET_STARTED": "Exportação GEE Asset iniciada",
    # Cache
    "CACHE_REMOVED": "Cache local removido",
    "CACHE_NOT_FOUND": "Cache não encontrado",
    # Misc
    "LOADING_TILES": "Carregando tiles...",
    "CONFIRM_DELETE": "Confirmar Exclusão",
    "CONFIRM_DELETE_ALL": "Confirmar Exclusão Total",
    "CANCELED": "Cancelado.",
    "GLOBAL_OPTS_SET": "Opções globais configuradas",
    # Dropdown labels
    "DROP_MODEL": "Modelo:",
    "DROP_REGION": "Região:",
    "DROP_YEAR": "Ano:",
    "DROP_TASK": "Tarefa:",
    "DROP_TASK_NAME": "Nome Tarefa:",
    # Hyperparams
    "HP_EPOCHS": "Épocas",
    "HP_PATIENCE": "Paciência",
    "HP_TEST_SPLIT": "Split de Teste",
    "HP_BALANCE": "Balanceamento",
    "HP_AUGMENT": "Aumento de Dados",
    "HP_ACTIVATION": "Função de Ativação",
    "HP_OPTIMIZER": "Otimizador",
    "HP_LOSS": "Função de Perda",
    # Extraction
    "EXTRACTION_TITLE": "Matriz de Extracao (Multissensor GCS)",
    "SAMPLE_SELECT": "Selecao de Amostras",
    "AVAILABLE": "Disponiveis",
    "SELECTED": "Selecionados",
    "ADD": "Adicionar",
    "REMOVE": "Remover",
    "CAMPAIGN": "Campanha",
    "SENSOR": "Sensor",
    "BANDS": "Bandas",
    "STATUS_OK": "OK",
    "STATUS_RUN": "Exec.",
    "STATUS_MISS": "Falt.",
    # Training
    "TRAINING_ID": "ID Treinamento:",
    "SHORTNAME": "Nome Rápido:",
    "COMMENTS": "Comentarios...",
    "START_TRAINING": "Iniciar Treinamento",
    "CANVAS_TITLE": "Centro de Treinamentos e Auditoria",
    "CANVAS_EMPTY": "Canvas vazio",
    "CANVAS_HINT": "Navegue e selecione modelos no painel lateral para visualiza-los aqui.",
    "LOSS_ACC": "Loss / Accuracy",
    "VIZ_METADATA": "Metadados",
    "VIZ_KPIS": "KPIs",
    "VIZ_CONFUSION": "Confusao",
    "VIZ_HISTORY": "Historico",
    "VIZ_PROB": "Prob",
    "VIZ_PR_CURVE": "PR-Curve",
    "VIZ_MANAGEMENT": "Gestao",
    # Repository
    "REPO_TITLE": "Ranking / Repositorio",
    "REPO_SEARCH": "Buscar no repositorio...",
    "REPO_SYNC": "Sincronizar GCS",
    "REPO_ALL": "Todos",
    "REPO_CLEAR": "Limpar",
    "REPO_SYNC_CATALOG": "Sincronizar Catalogo (GCS)",
    "REPO_SCANNING": "Escaneando GCS... Aguarde.",
    "REPO_SCAN_DONE": "Catálogo Sincronizado!",
    # Tab titles
    "SORT_BY": "Ordenar por:",
    "NEW_TRAINING": "Novo Treinamento",
    "TRAININGS": "Treinamentos",
    # M1 - Export
    "EXPORT_TITLE": "Exportação de Coleções",
    "EXPORT_SYNC": "Sincronizar Dados",
    "EXPORT_SELECT": "Selecionar Pendentes",
    "EXPORT_CLEAR": "Limpar Selecao",
    "EXPORT_START_BTN": "Iniciar Exportação",
    "EXPORT_SENT": "tarefas enviadas.",
    "EXPORT_NONE_SEL": "Nenhuma opção selecionada para exportação.",
    # M2 - Mosaic
    "MOSAIC_TITLE": "Montagem de Mosaicos",
    "MOSAIC_SYNC": "Sincronizar Dados",
    "MOSAIC_SELECT": "Selecionar Pendentes",
    "MOSAIC_CLEAR": "Limpar Seleção",
    "MOSAIC_START_BTN": "Iniciar Montagem",
    "MOSAIC_DONE": "Montagem concluída com sucesso",
    "M1_HEADER_TITLE": "M1 - Despachador",
    "M1_HEADER_SUBTITLE": "Interface multi-sensor para exportação",
    "M1_TITLE_FULL": "M1 - Despachador de Mosaicos",
    "M1_DESCRIPTION": "Interface multi-sensor para despachar composições (Assets/GCS) para a nuvem.",
    "LABEL_PROJECT": "Projeto",
    "BTN_SELECT_ROW": "[S]",
    "COL_TYPE": "Tipo",
    "M2_HEADER_TITLE": "M2 - Montador",
    "M2_HEADER_SUBTITLE": "Interface para montagem de mosaicos nacionais (COG)",
    "M2_CONSTRUCTOR_TITLE": "M2 - Montador de Mosaicos (COG)",
    "M2_CONSTRUCTOR_DESC": "Interface para converter chunks GCS em mosaicos COG nacionais.",
    "STATUS_READY": "PRONTO",
    # M0 - Startup
    "START_COUNTRY": "País",
    "START_BUCKET": "Bucket",
    "START_VERSION": "Versao",
    "START_SENSOR": "Sensor",
    "START_PERIODICITY": "Periodicidade",
    "START_CAMPAIGN": "Campanha",
    "START_FOUND": "encontrado",
    "START_NOT_FOUND": "não encontrado",
    "START_GDAL_OK": "GDAL encontrado e adicionado ao PATH",
    "START_GDAL_MISSING": "Aviso: Utilitários GDAL não encontrados",
    "START_COLAB_HINT": "No Google Colab, execute: !apt-get install -y gdal-bin",
    "START_WIN_HINT": "No Windows, certifique-se de que o GDAL esteja no seu PATH",
    # M6
    "M6_TITLE": "Processador Pós-Classificação",
    "M6_PRESET": "Usando Configuração Predefinida",
    "M6_DOWNLOAD": "Baixando fragmentos",
    "M6_VRT": "Construindo VRT",
    "M6_COG": "Convertendo para COG",
    "M6_UPLOAD": "Enviado",
    "M6_START": "Iniciando Filtragem",
    "M6_EXPORT_OK": "Tarefa de exportação iniciada",
    "M6_SUMMARY": "Resumo da Configuração",
    "TAB_ANALYTICS": "Analíticas",
    "TAB_M6_COVERAGE": "Cobertura",
    "GUIDE_M6_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>M6 - Mosaico, Stats e Publicação</h3>
        <p>Publica tiles classificados do M5: cria mosaicos regionais, calcula estatísticas de área queimada e envia para o Google Earth Engine.</p>
        <h4>Fluxo:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_publish}</b> — grupos de tiles classificados aguardando mosaico.</li>
            <li><b>{tab_done}</b> — regiões publicadas com mosaico e estatísticas.</li>
            <li><b>{tab_analytics}</b> — tabela consolidada de estatísticas com download.</li>
            <li><b>{tab_coverage}</b> — mapa de cobertura: publicado vs pendente.</li>
            <li>Execute <code>run_m6_publish()</code> no notebook para processar.</li>
        </ol>
    </div>""",
    "DOWNLOAD_TABLE": "Baixar Tabela",
    "ANALYTICS_FILTER_MODEL": "Modelo",
    "ANALYTICS_FILTER_REGION": "Região",
    "ANALYTICS_FILTER_PERIOD": "Período",
    "REFRESH_M6": "Atualizar M6",
    "M6_HEADER_TITLE": "M6 - Mosaico, Stats e Publicação",
    "M6_LABEL_PERIOD": "Período:",
    "M6_GROUPS_PENDING": "{n} grupos pendentes de mosaico",
    "M6_MOSAIC_OK": "mosaico OK",
    "M6_DOWNLOAD_MOSAIC": "OK \u2197",
    "M6_PUBLISHED_GROUPS": "{n} grupos publicados",
    "M6_BADGE_MOSAIC": "Mosaico",
    "M6_BADGE_STATS": "Estats",
    "M6_BADGE_GEE": "GEE",
    "M6_COL_PCT": "%",
    "M6_COL_HA": "ha",
    "M6_COL_TILES": "Tiles",
    "M6_LEGEND_PUBLISHED": "Publicado",
    "M6_LEGEND_PARTIAL": "Parcial",
    "M6_LEGEND_CLASSIFIED_ONLY": "Somente classificado",
    "M6_LEGEND_NO_DATA": "Sem dados",
    "M6_N_RECORDS": "{n} registros",
    "M6_DOWNLOAD_LINK": "Baixar {fname}",
    "M6_NO_STATS": "Nenhuma estatística consolidada. Execute M6 publish primeiro.",
    "M6_NO_MATCHING": "Nenhum registro correspondente.",
    "M6_NO_CLASSIFIED_GROUPS": "Nenhum grupo classificado encontrado.",
    # M7
    "M7_TITLE": "Curador",
    "M7_DESC": "Publicação de Coleção Pré-Oficial",
    "M7_PRESET": "Usando Votacao Predefinida",
    "M7_START": "Iniciando Curadoria",
    "M7_EXPORT_OK": "Exportação GEE Asset iniciada para",
    "M7_SUMMARY": "Resumo da Configuração",
    # M3
    "M3_TITLE": "M3 - Coleta de Amostras (GEE Toolkit Gateway)",
    "M3_SOURCE": "Acesso ao código fonte (GitHub)",
    "M3_EDITOR": "Acesso direto (Editor GEE)",
    "M3_DOCS": "Documentacao e normas de uso",
    "M3_SLIDES": "Apresentacao tutorial (Google Slides)",
    "M3_INTRO_LINE1": "A coleta de amostras e realizada exclusivamente",
    "M3_INTRO_LINE2": "através da interface JavaScript no Google Earth Engine.",
    # M4 - Analytics / KPIs
    "ACCURACY": "Acurácia",
    "PRECISION": "Precisão",
    "RECALL": "Recall",
    "F1_SCORE": "F1-Score",
    "AUTO_RATING": "Nota Automatica",
    "HUMAN_NOTE": "Nota Humana",
    "NO_METRICS": "Sem métricas",
    "CONFUSION_MATRIX": "Matriz de Confusão (%)",
    "HISTORICAL_EVOLUTION": "Evolução Histórica",
    "PROB_DISTRIBUTION": "Distribuição de Probabilidades",
    "CONFIDENCE": "Confiança",
    "LATENT_PROJ_2D": "Projeção Latente 2D",
    "COST_LOSS": "Custo (Loss)",
    "CLASSIC_METRICS": "Métricas Clássicas e Projeções Estáticas",
    "INTERACTIVE_LATENT": "Espaço Latente Interativo",
    "PCA_3D": "PCA 3D",
    "TSNE_3D": "t-SNE 3D",
    "PCA_3D_INTERACTIVE": "PCA 3D Interativo",
    "TSNE_3D_INTERACTIVE": "t-SNE 3D Interativo",
    "RETRAIN": "Retreinar",
    # M4 - Fire class labels
    "FIRE": "Fogo",
    "NO_FIRE": "Não-fogo",
    "FIRE_CLASS": "Fogo",
    "NO_FIRE_CLASS": "Não-fogo",
    # M4 - UI Labels
    "BASIC_STATS": "Estatísticas Básicas",
    "PCA_LATENT": "Espaço Latente PCA",
    "TSNE_LATENT": "Espaço Latente t-SNE",
    "VIZ_OPTIONS": "Opções de Visualização",
    "VIZ_PCA2D": "PCA 2D",
    "VIZ_PCA3D_STATIC": "PCA 3D (Estático)",
    "VIZ_PCA3D_INTERACTIVE": "PCA 3D (Interativo)",
    "VIZ_TSNE3D_STATIC": "t-SNE 3D (Estático)",
    "VIZ_TSNE3D_INTERACTIVE": "t-SNE 3D (Interativo)",
    "HYPERPARAMS_SECTION": "Hiperparâmetros (DNN)",
    "LIVE_TRAINING": "Treinamento ao Vivo",
    "TRAINING_IN_PROGRESS": "Treinamento em andamento...",
    "LIVE_TSNE_AUDIT": "Auditoria t-SNE ao Vivo (Espaço Latente)",
    "ADD_TO_CANVAS": "Adicionar ao Canvas",
    "REMOVE_FROM_CANVAS": "Remover do Canvas",
    "BTN_CLOSE": "X",
    "RELOAD_SAMPLES": "Recarregar lista de amostras do GCS",
    "PLUS": "+ ",
    # M4 - Analytics card labels
    "LAYERS_LABEL": "Camadas:",
    "LR_ABBR": "LR:",
    "SAMPLES_LABEL": "Amostras:",
    "NO_COMMENTS": "Sem comentários.",
    "HIDE_ADVANCED": "Ocultar parâmetros avançados",
    "SHOW_ALL_PARAMS": "Mostrar todos os parâmetros",
    "TSNE_AXIS_1": "t-SNE 1",
    "TSNE_AXIS_2": "t-SNE 2",
    "TSNE_AXIS_3": "t-SNE 3",
    "ACC_ABBR": "Acu",
    "F1_ABBR": "F1",
}

STRINGS_FR = {
    # General
    "LOADING": "Chargement...",
    "PROCESSING": "Traitement...",
    "DELETING": "Suppression...",
    "SAVING": "Enregistrement...",
    "SYNCING": "Synchronisation...",
    "SYNCING_TASKS": "Synchronisation des tâches GEE...",
    "SEARCHING": "Recherche...",
    "DONE": "Terminé",
    "ERROR": "Erreur",
    "SUCCESS": "Succès",
    "WARNING": "Attention",
    "INFO": "Information",
    "ALL": "Tous",
    "ALL_F": "Toutes",
    "NONE": "Aucun",
    "BACK": "Retour",
    "OK": "OK",
    "CONFIRM": "Confirmer",
    "CANCEL": "Annuler",
    "SAVE": "Enregistrer",
    "DELETE": "Supprimer",
    "SYNC": "Synchroniser",
    "SEARCH": "Rechercher",
    "CLEAR": "Effacer",
    "REFRESH": "Actualiser",
    "APPLY": "Appliquer",
    "CLOSE": "Fermer",
    "MODEL": "Modèle",
    "REGION": "Région",
    "YEAR": "Année",
    "PERIOD": "Période",
    "TASK": "Tâche",
    "TASK_NAME": "Nom Tâche",
    "STATUS": "Statut",
    "PROGRESS": "Progression",
    "DESCRIPTION": "Description",
    "NAME": "Nom",
    "DATE": "Date",
    "ID": "ID",
    # Empty states
    "NO_DATA": "Aucune donnée disponible.",
    "NO_RESULTS": "Aucun résultat trouvé.",
    "NO_TASKS": "Aucune tâche en attente.",
    "NO_TASKS_PUBLISH": "Aucune tâche prête à publier.",
    "NO_TASKS_DONE": "Aucune tâche terminée.",
    "NO_TILES_GCS": "Aucun tile dans GCS.",
    "NO_MAP": "Impossible de générer la carte. Vérifier connexion GEE.",
    "NO_SELECTION": "Aucune option sélectionnée.",
    "NO_SAMPLES": "Aucun échantillon trouvé avec ce filtre.",
    "NO_COGS": "Aucun COG trouvé dans le dépôt GCS.",
    # M5 Widgets
    "ADD_BATCH": "Ajouter un lot au plan",
    "REFRESH_VIEW": "Actualiser la vue",
    "LOAD_TO_QUEUE": "Charger dans le plan",
    "CLEAR_TEMP_TASKS": "Effacer les tâches temporaires",
    "SAVE_TASK_GCS": "Enregistrer tâche GCS",
    "EXCLUDE_TASK_GCS": "Exclure tâche GCS",
    "DELETE_MODEL": "Supprimer le modèle",
    "DISCARD_WORKPLAN": "Abandonner le Plan de Travail",
    "GLOBAL_ACTIONS": "Actions Globales",
    "NO_PENDING_JOBS": "Aucun travail en attente à classer.",
        "ERR_NO_SAMPLES": "Erreur : Aucun échantillon sélectionné.",
        "ERR_NO_BANDS": "Erreur : Aucune bande sélectionnée dans la Matrice d'Extraction.",
        "ERR_INCOMPATIBLE_ENV": "Environnement local incompatible",
        "ERR_NO_AVX": "Votre CPU ne supporte pas les instructions AVX/AVX2 requises par TensorFlow.",
        "ERR_RUN_ON_COLAB": "Veuillez exécuter cet entraînement sur Google Colab.",
    "CARD_SAVED": "Enregistré \u2713",
    "CARD_TEMP": "Temporaire",
    "CARD_SAVED_PARTIAL": "{s}/{t} Enregistrés",
    "BTN_DETAILS_COLLAPSED": "Détails \u25bc",
    "BTN_DETAILS_EXPANDED": "Détails \u25b2",
    "NO_METADATA": "Métadonnées du modèle indisponibles (hors ligne ou metadata.json manquant).",
    "DELETE_REGION": "Supprimer la région",
    "DELETE_SELECTED": "Supprimer sélectionnés",
    "DELETE_ALL": "Tout supprimer",
    "DELETE_JOB": "Tout supprimer",
    "VIEW_TILES": "Voir les tiles",
    "HIDE_TILES": "Cacher",
    "REFRESH_MAP": "Actualiser la carte",
    "TASK_NAME_PLACEHOLDER": "Ex: Classifier Amazonie Basse 2025 (Lucas)",
    # M5 - Section titles
    "SECTION_MODEL": "1. Sélectionnez Modèle (unique) :",
    "SECTION_REGIONS": "2. Sélectionnez Régions :",
    "SECTION_PERIODS": "3. Sélectionnez Périodes (Année / Année_Mois) :",
    "LOADING_CAMPAIGNS": "Chargement campagnes...",
    "CAMPAIGN_LABEL": "Campagne :",
    "FILTER_LABEL": "Filtrer :",
    "FILTER_PLACEHOLDER": "rechercher par modèle, région, période...",
    # M5 - Region diagnostic
    "ERR_CONFIG_ASSET_REGIONS": "CONFIG[asset_regions] non défini",
    "ERR_ASSET_NOT_FOUND": "Asset INTROUVABLE dans GEE :",
    "ERR_ASSET_EMPTY": "L'asset existe mais est VIDE :",
    "ERR_ASSET_COLUMN": "Colonne <code>{prop}</code> introuvable",
    "ERR_GEE_ACCESS": "Erreur d'accès à GEE :",
        "WARN_REGION_FALLBACK": "Impossible de peupler les régions automatiquement",
        "ERR_RENDER_PENDING": "Erreur lors du rendu des tâches en attente:",
    "COLUMN_USED": "Colonne utilisée :",
    # M5 - Validation
    "ERR_NO_TASK_NAME": "Attention : Vous devez attribuer un Nom de Tâche.",
    "ERR_NO_SELECTION": "Attention : Sélectionnez 1 Modèle et au moins 1 Région et 1 Période.",
    # M5 - Feedback messages
    "SAVING_TO_GCS": "Enregistrement dans GCS...",
    "MSG_NO_TASKS_ADDED": "Aucune tâche ajoutée.",
    "MSG_TASKS_ADDED": "Succès : {n} tâche(s) ajoutée(s).",
    "FAIL_SAVE_LOCAL": "échec de la sauvegarde locale (verrou occupé)",
    "FAIL_GCS_UPLOAD": "{n} échec(s) sur GCS",
    "MSG_GCS_OK": "{n} enregistrée(s) dans GCS.",
    "MSG_SKIPPED_QUEUED": "{n} ignorée(s) (déjà dans la file).",
    "WARN_ALREADY_QUEUED": "Attention : {n} tâches déjà dans la file.",
    "MSG_LOADED": "{loaded} chargée(s), {skipped} ignorée(s).",
    "TAREA_SAVED": "Tâche enregistrée dans GCS pour {m}.",
    "TAREA_DELETED": "Tâche supprimée de GCS : {m}.",
    "WORKPLAN_DISCARDED": "Plan abandonné : {m} ({n} travaux supprimés).",
    "QUEUE_CLEARED": "File vidée (GCS supprimé).",
    "SAVED_TO_GCS": "{n} travail(aux) enregistré(s) dans GCS.",
    "ALREADY_SAVED": "{n} travail(aux) déjà enregistré(s).",
    "TEMP_JOBS_REMOVED": "{n} travail(aux) temporaire(s) supprimé(s).",
    "NO_TEMP_JOBS": "Aucun travail temporaire à supprimer.",
    "REMOVED_FROM_GCS": "{n} travail(aux) supprimé(s) de GCS.",
    "NO_TILES_GCS_ALT": "Aucun tile dans GCS.",
    "TILES_DELETED": "{n} tile(s) supprimé(s).",
    "SELECT_TILES_PROMPT": "Sélectionnez les tiles à supprimer.",
    "SYNC_FROM_GCS": "{n} travail(aux) synchronisé(s) de GCS.",
    # M5 - Tareas section
    "TAREAS_HEADER": "Tâches dans GCS ({n} disponibles)",
    "TAREA_SUMMARY": "Modèle : {m} | Régions : {r} | Périodes : {p}",
    # M5 - Scale bar
    "SCALE_BAR_WIDTH": "~{w} km de large",
    # M5 - Pending debug
    "INFO_FILTER_COUNT": "{total} en attente | {filtered} après filtre",
    "GRID_COUNT": "({n} cellules)",
    # M5 - Map legend
    "MAP_LEGEND_COUNTRY": "Pays",
    "MAP_LEGEND_REGIONS": "Régions",
    "MAP_LEGEND_GRID": "Grille cim-world",
    # M5 - Model inference errors
    "ERR_MODEL_NO_METADATA": "metadata.json introuvable dans {path}",
    "ERR_MODEL_NO_BANDS_CONFIG": "Le modèle n'a pas 'bands_config' dans les métadonnées.",
    "ERR_MODEL_NO_BAND_ORDER": "Le modèle n'a pas 'band_order' dans les métadonnées. Réentraînez avec le M4 actuel (commit 9b95392+).",
    "ERR_MODEL_INCONSISTENCY": "Incohérence dans les métadonnées du modèle : {details}",
    # M5 - Jobs details panel
    "JOBS_REGISTERED": "Travaux enregistrés",
    "JOBS_SUMMARY": "{total} travail(aux), {regions} région(s), {periods} période(s)",
    "JOB_REGION": "Région",
    "JOB_PERIOD": "Période",
    "JOB_STATUS": "Statut",
    "JOB_TASK": "Tâche",
    # M5 - Model metadata panel
    "META_SENSORS": "Capteurs",
    "META_PERIODICITY": "Périodicité",
    "META_BANDS_INPUT": "Bandas (entrée)",
    "META_ITERATIONS": "Itérations",
    "META_LEARNING_RATE": "Learning rate",
    "META_CAMPAIGN": "Campagne",
    # M5 Tabs
    "TAB_GUIDE": "Guide",
    "TAB_REGISTER": "Enregistrer",
    "TAB_PENDING": "En attente",
    "TAB_PUBLISH": "À publier",
    "TAB_MAP": "Carte",
    "TAB_DONE": "Terminées",
    "GUIDE_M5_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:5px;'>M5 - Classification Régionale à Grande Échelle</h3>
        <p>Classifie plusieurs régions (grille cim-world-1-250000) en utilisant les modèles M4.</p>
        <h4>Flux :</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_register}</b> — sélectionnez modèle + régions + périodes.</li>
            <li><b>{tab_pending}</b> — suivez la classification tile par tile.</li>
            <li><b>{tab_map}</b> — aperçu général du traitement en direct.</li>
            <li>Exécutez <code>run_m5_workplan()</code> dans le notebook pour traiter.</li>
        </ol>
        <h4>Après la Classification (M6) :</h4>
        <p>Utilisez <b>M6</b> pour mosaïquer les tiles, générer des statistiques régionales et publier dans GEE.
        Ouvrez l'interface M6 et exécutez <code>run_m6_publish()</code> dans une cellule séparée.</p>
        <p>Les travaux classifiés (COMPLETED) apparaissent automatiquement dans M6.</p>
    </div>""",
    "GUIDE_M1_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <div style='margin-bottom:15px;padding:10px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px;font-size:13px;'>
            <b>Remarque :</b> Actuellement, seules les compositions mensuelles pour <b>Sentinel-2</b> avec <b>minnbr</b> et <b>minnbr_buffer</b> sont publiées. D'autres combinaisons capteur/période/mosaïque peuvent être expérimentales.
        </div>
        <h3 style='color:#2c3e50; border-bottom:2px solid #e67e22; padding-bottom:5px;'>Module 1 (M1) — Exportation GEE Multi-Capteur</h3>
        <p>M1 exporte des mosaïques satellitaires sans nuage depuis Google Earth Engine vers Google Cloud Storage (GCS) ou des Assets GEE.
        Supporte LANDSAT 5/7, LANDSAT 8/9, Sentinel-2, HLS et MODIS avec corrections radiometriques et masquage des nuages specifiques par capteur.</p>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:15px;'>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Capteurs et Sources</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>LANDSAT 5/7</b> — archive historique (1984–)</li>
                    <li><b>LANDSAT 8/9</b> — OLI/TIRS actuel avec bandes thermales</li>
                    <li><b>Sentinel-2</b> — 10m de resolution, revisite 5 jours</li>
                    <li><b>HLS</b> — produit harmonise LANDSAT/Sentinel</li>
                    <li><b>MODIS</b> — couverture globale quotidienne (250m–1km)</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Methodes de Mosaique</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>minnbr</b> — moins de nuages selon le classement NBR</li>
                    <li><b>minndvi</b> — moins de nuages selon le classement NDVI</li>
                    <li><b>median</b> — composition mediane pixel par pixel</li>
                    <li><b>minnbr_buffer</b> — minnbr avec masque tampon de feu INPE</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Flux de Travail</h4>
                <ol style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Sélectionnez l onglet du capteur (LANDSAT, S2, HLS, MODIS)</li>
                    <li>Choisissez la periode (mensuelle / annuelle)</li>
                    <li>Sélectionnez la methode de mosaique</li>
                    <li>Cochez les cellules de date et de bande</li>
                    <li>Cliquez sur <b>Démarrer Exportation</b></li>
                </ol>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Informations Techniques</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Correction radiometrique par capteur</li>
                    <li>Masquage des nuages via QA_PIXEL, Fmask, CS+</li>
                    <li>Masque tampon des incendies INPE</li>
                    <li>Sortie: fragments GeoTIFF dans GCS / ImageCollection dans GEE</li>
                </ul>
            </div>
        </div>
        <div style='margin-top:15px;padding:10px;background:#fef3e2;border-left:4px solid #e67e22;border-radius:4px;font-size:13px;'>
            <b>Conseil:</b> Utilisez <b>Synchroniser Donnees</b> pour actualiser le cache. Utilisez <b>Selectionner en Attente</b> pour cocher automatiquement toutes les dates disponibles.
        </div>
    </div>""",
    "GUIDE_M2_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>Module 2 (M2) — Assemblage de Mosaiques COG</h3>
        <p>M2 assemble les fragments GeoTIFF exportes par M1 en COGs (Cloud-Optimized GeoTIFF) nationaux complets en utilisant GDAL.</p>
        <h4>Flux:</h4>
        <ol style='line-height:1.6;'>
            <li><b>Verifiez le statut</b> — OK (COG existe), READY (fragments disponibles), MISS (rien de disponible)</li>
            <li><b>Sélectionnez les cellules</b> — cochez les bandes a assembler</li>
            <li>Cliquez sur <b>Démarrer Assemblage</b></li>
            <li>GDAL telecharge les fragments → construit VRT → convertit en COG (LZW) → upload vers GCS</li>
        </ol>
        <h4>Exigences:</h4>
        <ul style='line-height:1.6;'>
            <li>GDAL installe (<code>gdalbuildvrt</code>, <code>gdal_translate</code>)</li>
            <li>M1 doit avoir ete execute d abord pour produire les fragments source</li>
        </ul>
        <div style='margin-top:15px;padding:10px;background:#e8f8ed;border-left:4px solid #27ae60;border-radius:4px;font-size:13px;'>
            <b>Conseil:</b> Les COGs sont stockes dans <code>.../COG/</code> et consommes par M4 (entrainement) et M5 (classification).
        </div>
    </div>""",
    # Map / Grid
    "LIVE_PROCESSING": "Traitement en direct",
    "CURRENT_TILE": "Tile actuel",
    "COMPLETED": "Terminés",
    "GRID_REGION": "Région",
    "GRID_CELLS": "Cellules cim-world",
    # Tiles / GCS
    "TILES": "tiles",
    "MOSAIC": "mosaique",
    "STATS": "stats",
    "SELECT_TILES_DELETE": "Sélectionnez les tiles a supprimer.",
    "TILES_REMOVED": "tiles supprimés.",
    # Task status
    "STATUS_QUEUED": "Dans le plan",
    "STATUS_RUNNING": "En cours",
    "STATUS_COMPLETED": "Complété",
    "STATUS_FINISHED": "Terminé",
    "STATUS_ERROR": "Erreur",
    "STATUS_PUBLISHING": "Publication",
    "STATUS_PUBLISHED": "Publié",
    "STATUS_SKIPPED": "Ignoré",
    # M4 - Training
        "MODEL_TRAINER": "Entraîneur de modèle",
        "M4_HEADER_TITLE": "M4 - Entraîneur de modèle",
        "SELECTED_OK": "[OK] Sélectionnés",
        "TRAINING_PROGRESS": "Entraînement:",
    "ITERATIONS": "Itérations",
    "BATCH_SIZE": "Taille du lot",
    "LEARNING_RATE": "Taux d'apprentissage",
    "HIDDEN_LAYERS": "Couches cachées",
    "ACTIVATION": "Activation",
    "DROPOUT": "Dropout",
    "OPTIMIZER": "Optimiseur",
    "LOSS_FN": "Fonction de perte",
    "METRICS": "Métriques",
    "SAMPLE_SELECTION": "Sélection d'échantillons",
    "EXTRACTION_MATRIX": "Matrice d'extraction",
    "MODEL_CONFIG": "Configuration du modèle",
    "GCS_DEST": "Destination GCS",
    "USAGE_GUIDE": "Guide d'utilisation",
    "GUIDE_M4_HTML": """<div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; background: #fdfdfd; color: #2c3e50; line-height: 1.6;">
        <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">M4 Model Trainer - Guide d Utilisation</h1>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 20px;">
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #3498db; margin-top:0;">Structure de la Plateforme</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>{usage_guide}:</b> Ecran d orientation et de documentation.</li>
                    <li><b>{new_training}:</b> Configuration de nouvelles experiences, selection d échantillons et de bandes.</li>
                    <li><b>{trainings}:</b> Classement historique avec metriques detaillees et gestion des modèles.</li>
                    <li><b>Canvas:</b> Bureau d audit parallele pour comparer plusieurs modèles en profondeur.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #9b59b6; margin-top:0;">Concepts Techniques</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>TensorFlow:</b> Moteur IA de Google pour les calculs mathematiques massifs.</li>
                    <li><b>DNN (Deep Neural Network):</b> Reseau profond qui imite l apprentissage humain.</li>
                    <li><b>Neurones:</b> Unites qui traitent les signaux et activent les modèles d apprentissage.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #e67e22; margin-top:0;">Hyperparametres (DNN)</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Layers:</b> Architecture du reseau. Plus de couches capturent des details plus fins.</li>
                    <li><b>Learning Rate (LR):</b> Controle la vitesse d ajustement du modèle.</li>
                    <li><b>Epochs:</b> Cycles d entrainement complets sur l ensemble d échantillons.</li>
                    <li><b>Batch Size:</b> Blocs de données traites avant chaque mise a jour.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #27ae60; margin-top:0;">Dictionnaire de Qualite</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Accuracy:</b> Pourcentage total de reussites globales.</li>
                    <li><b>Precision:</b> Fidelite: Quelle part du feu marque est reelle? (Evite les faux positifs).</li>
                    <li><b>Recall:</b> Couverture: Quelle part du feu reel a ete trouvée? (Evite les omissions).</li>
                    <li><b>F1-Score:</b> Moyenne harmonique. Le meilleur equilibre entre Precision et Recall.</li>
                    <li><b>Note IA:</b> Audit automatique qui penalise severement les omissions.</li>
                    <li><b>Note Humaine:</b> Evaluation subjective (1-5) sur l Espace Latent.</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 30px; padding: 15px; background: #e8f4fd; border-left: 5px solid #3498db; border-radius: 4px; font-size:14px;">
            <b>[Conseil] Pro-Tip de l Auditeur:</b> Utilisez le <b>Canvas</b> pour charger un ancien modèle (benchmark) et votre nouveau modèle. Comparez si la separation des classes en t-SNE 3D s est amelioree ou s il y a de nouvelles zones de confusion.
        </div>
    </div>""",
    # M4 - Canvas
    "METADATA": "Métadonnées",
    "KPIS": "KPIs",
    "CONFUSION": "Confusion",
    "HISTORY": "Historique",
    "PROB": "Prob",
    "PR_CURVE": "Courbe PR",
    "MANAGEMENT": "Gestion",
    "RANKING": "Classement / Dépôt",
    "SELECTED_CANVAS": "Sélectionnés dans Canvas",
    "SEARCH_REPO": "Rechercher dans le dépôt...",
    "SEARCH_SAMPLES": "Rechercher échantillons...",
    "SYNC_CATALOG": "Synchroniser catalogue (GCS)",
    "APPLY_VISIBILITY": "Appliquer la visibilité",
    # M1 / M2
    "SYNC_DATA": "Synchroniser données",
    "SELECT_PENDING": "Sélectionner en attente",
    "CLEAR_SELECTION": "Effacer la selection",
    "EXPORT_START": "Démarrer exportation",
    "MOSAIC_START": "Démarrer assemblage",
    # M6
    "POST_CLASSIFICATION": "Processeur post-classification",
    "USING_PRESET": "Utilisation configuration predefinie",
    "FILTER_START": "Démarrer filtrage",
    "EXPORT_TASK_STARTED": "Tâche d'exportation démarrée",
    "CONFIG_SUMMARY": "Résumé de configuration",
    # M7
    "CURATOR": "Curateur",
    "CURATOR_DESC": "Publication collection pre-officielle",
    "USING_VOTES_PRESET": "Utilisation vote prédéfini",
    "CURATION_START": "Démarrer curation",
    "EXPORT_ASSET_STARTED": "Exportation GEE Asset demarree",
    # Cache
    "CACHE_REMOVED": "Cache local supprimé",
    "CACHE_NOT_FOUND": "Cache introuvable",
    # Misc
    "LOADING_TILES": "Chargement tiles...",
    "CONFIRM_DELETE": "Confirmer la suppression",
    "CONFIRM_DELETE_ALL": "Confirmer la suppression totale",
    "CANCELED": "Annulé.",
    "GLOBAL_OPTS_SET": "Options globales configurées",
    # Dropdown labels
    "DROP_MODEL": "Modèle:",
    "DROP_REGION": "Région:",
    "DROP_YEAR": "Année:",
    "DROP_TASK": "Tâche:",
    "DROP_TASK_NAME": "Nom Tâche:",
    # Hyperparams
    "HP_EPOCHS": "Époques",
    "HP_PATIENCE": "Patience",
    "HP_TEST_SPLIT": "Split de test",
    "HP_BALANCE": "Équilibrage",
    "HP_AUGMENT": "Augmentation données",
    "HP_ACTIVATION": "Fonction d'activation",
    "HP_OPTIMIZER": "Optimiseur",
    "HP_LOSS": "Fonction de perte",
    # Extraction
    "EXTRACTION_TITLE": "Matrice d'extraction (Multicapteur GCS)",
    "SAMPLE_SELECT": "Sélection échantillons",
    "AVAILABLE": "Disponibles",
    "SELECTED": "Sélectionnés",
    "ADD": "Ajouter",
    "REMOVE": "Retirer",
    "CAMPAIGN": "Campagne",
    "SENSOR": "Capteur",
    "BANDS": "Bandes",
    "STATUS_OK": "OK",
    "STATUS_RUN": "Exec.",
    "STATUS_MISS": "Manq.",
    # Training
    "TRAINING_ID": "ID Entraînement:",
    "SHORTNAME": "Nom court:",
    "COMMENTS": "Commentaires...",
    "START_TRAINING": "Démarrer l'entraînement",
    "CANVAS_TITLE": "Centre d'entraînement et d'audit",
    "CANVAS_EMPTY": "Canvas vide",
    "CANVAS_HINT": "Parcourez et sélectionnez des modèles dans le panneau latéral.",
    "LOSS_ACC": "Perte / Précision",
    "VIZ_METADATA": "Metadonnées",
    "VIZ_KPIS": "KPIs",
    "VIZ_CONFUSION": "Confusion",
    "VIZ_HISTORY": "Historique",
    "VIZ_PROB": "Prob",
    "VIZ_PR_CURVE": "Courbe PR",
    "VIZ_MANAGEMENT": "Gestion",
    # Repository
    "REPO_TITLE": "Classement / Depot",
    "REPO_SEARCH": "Rechercher dans le dépôt...",
    "REPO_SYNC": "Synchroniser GCS",
    "REPO_ALL": "Tous",
    "REPO_CLEAR": "Effacer",
    "REPO_SYNC_CATALOG": "Synchroniser catalogue (GCS)",
    "REPO_SCANNING": "Analyse GCS... Patientez.",
    "REPO_SCAN_DONE": "Catalogue synchronisé!",
    # Tab titles
    "SORT_BY": "Trier par:",
    "NEW_TRAINING": "Nouvel entraînement",
    "TRAININGS": "Entraînements",
    # M1 - Export
    "EXPORT_TITLE": "Exportation de collections",
    "EXPORT_SYNC": "Synchroniser données",
    "EXPORT_SELECT": "Selectionner en attente",
    "EXPORT_CLEAR": "Effacer selection",
    "EXPORT_START_BTN": "Démarrer exportation",
    "EXPORT_SENT": "tâches envoyées.",
    "EXPORT_NONE_SEL": "Aucune option sélectionnée pour l'exportation.",
    # M2 - Mosaic
    "MOSAIC_TITLE": "Assemblage de mosaïques",
    "MOSAIC_SYNC": "Synchroniser données",
    "MOSAIC_SELECT": "Sélectionner en attente",
    "MOSAIC_CLEAR": "Effacer sélection",
    "MOSAIC_START_BTN": "Démarrer assemblage",
    "MOSAIC_DONE": "Assemblage terminé avec succès",
    "M1_HEADER_TITLE": "M1 - Répartiteur",
    "M1_HEADER_SUBTITLE": "Interface multisensor d'exportation",
    "M1_TITLE_FULL": "M1 - Répartiteur de Mosaïques",
    "M1_DESCRIPTION": "Interface multisensor pour répartir des compositions (Assets/GCS) vers le cloud.",
    "LABEL_PROJECT": "Projet",
    "BTN_SELECT_ROW": "[S]",
    "COL_TYPE": "Type",
    "M2_HEADER_TITLE": "M2 - Assembleur",
    "M2_HEADER_SUBTITLE": "Interface d'assemblage de mosaïques COG nationales",
    "M2_CONSTRUCTOR_TITLE": "M2 - Assembleur de Mosaïques (COG)",
    "M2_CONSTRUCTOR_DESC": "Interface pour convertir des fragments GCS en mosaïques COG nationales.",
    "STATUS_READY": "PRÊT",
    # M0 - Startup
    "START_COUNTRY": "Pays",
    "START_BUCKET": "Bucket",
    "START_VERSION": "Version",
    "START_SENSOR": "Capteur",
    "START_PERIODICITY": "Périodicité",
    "START_CAMPAIGN": "Campagne",
    "START_FOUND": "trouvé",
    "START_NOT_FOUND": "non trouvé",
    "START_GDAL_OK": "GDAL trouvé et ajouté au PATH",
    "START_GDAL_MISSING": "Avertissement: Utilitaires GDAL non trouvés",
    "START_COLAB_HINT": "Sur Google Colab, exécutez: !apt-get install -y gdal-bin",
    "START_WIN_HINT": "Sur Windows, assurez-vous que GDAL est dans votre PATH",
    # M6
    "M6_TITLE": "Processeur post-classification",
    "M6_PRESET": "Utilisation configuration prédéfinie",
    "M6_DOWNLOAD": "Téléchargement fragments",
    "M6_VRT": "Construction VRT",
    "M6_COG": "Conversion en COG",
    "M6_UPLOAD": "Téléchargé",
    "M6_START": "Démarrage filtrage",
    "M6_EXPORT_OK": "Tâche d'exportation démarrée",
    "M6_SUMMARY": "Résumé configuration",
    "TAB_ANALYTICS": "Analytiques",
    "TAB_M6_COVERAGE": "Couverture",
    "GUIDE_M6_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>M6 - Mosaïque, Stats et Publication</h3>
        <p>Publie les tuiles classifiées de M5 : crée des mosaïques régionales, calcule les statistiques de surface brûlée et télécharge vers Google Earth Engine.</p>
        <h4>Flux :</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_publish}</b> — groupes de tuiles classifiées en attente de mosaïque.</li>
            <li><b>{tab_done}</b> — régions publiées avec mosaïque et statistiques.</li>
            <li><b>{tab_analytics}</b> — tableau consolidé des statistiques avec téléchargement.</li>
            <li><b>{tab_coverage}</b> — carte de couverture : publié vs en attente.</li>
            <li>Exécutez <code>run_m6_publish()</code> dans le notebook pour traiter.</li>
        </ol>
    </div>""",
    "DOWNLOAD_TABLE": "Télécharger Tableau",
    "ANALYTICS_FILTER_MODEL": "Modèle",
    "ANALYTICS_FILTER_REGION": "Région",
    "ANALYTICS_FILTER_PERIOD": "Période",
    "REFRESH_M6": "Actualiser M6",
    "M6_HEADER_TITLE": "M6 - Mosaïque, Stats et Publication",
    "M6_LABEL_PERIOD": "Période:",
    "M6_GROUPS_PENDING": "{n} groupes en attente de mosaïque",
    "M6_MOSAIC_OK": "mosaïque OK",
    "M6_DOWNLOAD_MOSAIC": "OK \u2197",
    "M6_PUBLISHED_GROUPS": "{n} groupes publiés",
    "M6_BADGE_MOSAIC": "Mosaïque",
    "M6_BADGE_STATS": "Stats",
    "M6_BADGE_GEE": "GEE",
    "M6_COL_PCT": "%",
    "M6_COL_HA": "ha",
    "M6_COL_TILES": "Tuiles",
    "M6_LEGEND_PUBLISHED": "Publié",
    "M6_LEGEND_PARTIAL": "Partiel",
    "M6_LEGEND_CLASSIFIED_ONLY": "Classifié uniquement",
    "M6_LEGEND_NO_DATA": "Pas de données",
    "M6_N_RECORDS": "{n} enregistrements",
    "M6_DOWNLOAD_LINK": "Télécharger {fname}",
    "M6_NO_STATS": "Aucune statistique consolidée. Exécutez M6 publish d'abord.",
    "M6_NO_MATCHING": "Aucun enregistrement correspondant.",
    "M6_NO_CLASSIFIED_GROUPS": "Aucun groupe classifié trouvé.",
    # M7
    "M7_TITLE": "Curateur",
    "M7_DESC": "Publication collection pré-officielle",
    "M7_PRESET": "Utilisation vote prédéfini",
    "M7_START": "Démarrage curation",
    "M7_EXPORT_OK": "Exportation GEE Asset démarrée pour",
    "M7_SUMMARY": "Résumé configuration",
    # M3
    "M3_TITLE": "M3 - Collecte d'échantillons (Passerelle GEE Toolkit)",
    "M3_SOURCE": "Code source (GitHub)",
    "M3_EDITOR": "Accès direct (Éditeur GEE)",
    "M3_DOCS": "Documentation et normes d'utilisation",
    "M3_SLIDES": "Presentation tutoriel (Google Slides)",
    "M3_INTRO_LINE1": "La collecte d'echantillons est effectuee exclusivement",
    "M3_INTRO_LINE2": "via l'interface JavaScript dans Google Earth Engine.",
    # M4 - Analytics / KPIs
    "ACCURACY": "Précision",
    "PRECISION": "Précision",
    "RECALL": "Rappel",
    "F1_SCORE": "F1-Score",
    "AI_NOTE": "Note IA",
    "HUMAN_NOTE": "Note Humaine",
    "NO_METRICS": "Aucune métrique",
    "CONFUSION_MATRIX": "Matrice de Confusion (%)",
    "HISTORICAL_EVOLUTION": "Évolution Historique",
    "PROB_DISTRIBUTION": "Distribution des Probabilités",
    "CONFIDENCE": "Confiance",
    "LATENT_PROJ_2D": "Projection Latente 2D",
    "COST_LOSS": "Coût (Loss)",
    "CLASSIC_METRICS": "Métriques Classiques et Projections Statiques",
    "INTERACTIVE_LATENT": "Espace Latent Interactif",
    "PCA_3D": "PCA 3D",
    "TSNE_3D": "t-SNE 3D",
    "PCA_3D_INTERACTIVE": "PCA 3D Interactif",
    "TSNE_3D_INTERACTIVE": "t-SNE 3D Interactif",
    "RETRAIN": "Réentraîner",
    # M4 - Fire class labels
    "FIRE": "Feu",
    "NO_FIRE": "Non-feu",
    "FIRE_CLASS": "Feu",
    "NO_FIRE_CLASS": "Non-feu",
    # M4 - UI Labels
    "BASIC_STATS": "Statistiques de Base",
    "PCA_LATENT": "Espace Latent PCA",
    "TSNE_LATENT": "Espace Latent t-SNE",
    "VIZ_OPTIONS": "Options de Visualisation",
    "VIZ_PCA2D": "PCA 2D",
    "VIZ_PCA3D_STATIC": "PCA 3D (Statique)",
    "VIZ_PCA3D_INTERACTIVE": "PCA 3D (Interactif)",
    "VIZ_TSNE3D_STATIC": "t-SNE 3D (Statique)",
    "VIZ_TSNE3D_INTERACTIVE": "t-SNE 3D (Interactif)",
    "HYPERPARAMS_SECTION": "Hyperparamètres (DNN)",
    "LIVE_TRAINING": "Entraînement en Direct",
    "TRAINING_IN_PROGRESS": "Entraînement en cours...",
    "LIVE_TSNE_AUDIT": "Audit t-SNE en Direct (Espace Latent)",
    "ADD_TO_CANVAS": "Ajouter au Canvas",
    "REMOVE_FROM_CANVAS": "Retirer du Canvas",
    "BTN_CLOSE": "X",
    "RELOAD_SAMPLES": "Recharger la liste d'échantillons du GCS",
    # M4 - Analytics card labels
    "LAYERS_LABEL": "Couches:",
    "LR_ABBR": "LR:",
    "SAMPLES_LABEL": "Échantillons:",
    "NO_COMMENTS": "Aucun commentaire.",
    "HIDE_ADVANCED": "Masquer les paramètres avancés",
    "SHOW_ALL_PARAMS": "Afficher tous les paramètres",
    "TSNE_AXIS_1": "t-SNE 1",
    "TSNE_AXIS_2": "t-SNE 2",
    "TSNE_AXIS_3": "t-SNE 3",
    "ACC_ABBR": "Préc",
    "F1_ABBR": "F1",
}

STRINGS_ID = {
    # General
    "LOADING": "Memuat...",
    "PROCESSING": "Memproses...",
    "DELETING": "Menghapus...",
    "SAVING": "Menyimpan...",
    "SYNCING": "Menyinkronkan...",
    "SYNCING_TASKS": "Menyinkronkan tugas GEE...",
    "SEARCHING": "Mencari...",
    "DONE": "Selesai",
    "ERROR": "Kesalahan",
    "SUCCESS": "Berhasil",
    "WARNING": "Peringatan",
    "INFO": "Informasi",
    "ALL": "Semua",
    "ALL_F": "Semua",
    "NONE": "Tidak Ada",
    "BACK": "Kembali",
    "OK": "OK",
    "CONFIRM": "Konfirmasi",
    "CANCEL": "Batal",
    "SAVE": "Simpan",
    "DELETE": "Hapus",
    "SYNC": "Sinkron",
    "SEARCH": "Cari",
    "CLEAR": "Bersihkan",
    "REFRESH": "Muat Ulang",
    "APPLY": "Terapkan",
    "CLOSE": "Tutup",
    "MODEL": "Model",
    "REGION": "Wilayah",
    "YEAR": "Tahun",
    "PERIOD": "Periode",
    "TASK": "Tugas",
    "TASK_NAME": "Nama Tugas",
    "STATUS": "Status",
    "PROGRESS": "Progres",
    "DESCRIPTION": "Deskripsi",
    "NAME": "Nama",
    "DATE": "Tanggal",
    "ID": "ID",
    # Empty states
    "NO_DATA": "Tidak ada data tersedia.",
    "NO_RESULTS": "Tidak ada hasil ditemukan.",
    "NO_TASKS": "Tidak ada tugas tertunda.",
    "NO_TASKS_PUBLISH": "Tidak ada tugas siap dipublikasikan.",
    "NO_TASKS_DONE": "Belum ada tugas selesai.",
    "NO_TILES_GCS": "Tidak ada tile di GCS.",
    "NO_MAP": "Tidak dapat membuat peta. Periksa koneksi GEE.",
    "NO_SELECTION": "Tidak ada opsi dipilih.",
    "NO_SAMPLES": "Tidak ada sampel ditemukan dengan filter ini.",
    "NO_COGS": "Tidak ada COG ditemukan di repositori GCS.",
    # M5 Widgets
    "ADD_BATCH": "Tambah Batch ke Rencana",
    "REFRESH_VIEW": "Muat Ulang Tampilan",
    "LOAD_TO_QUEUE": "Muat ke Rencana",
    "CLEAR_TEMP_TASKS": "Bersihkan Tugas Sementara",
    "SAVE_TASK_GCS": "Simpan Tugas GCS",
    "EXCLUDE_TASK_GCS": "Keluarkan Tugas GCS",
    "DELETE_MODEL": "Hapus Model",
    "DISCARD_WORKPLAN": "Buang Rencana Kerja",
    "GLOBAL_ACTIONS": "Aksi Global",
    "NO_PENDING_JOBS": "Tidak ada pekerjaan tertunda untuk diklasifikasikan.",
        "ERR_NO_SAMPLES": "Error: Tidak ada sampel yang dipilih.",
        "ERR_NO_BANDS": "Error: Tidak ada pita yang dipilih di Matriks Ekstraksi.",
        "ERR_INCOMPATIBLE_ENV": "Lingkungan lokal tidak kompatibel",
        "ERR_NO_AVX": "CPU Anda tidak mendukung instruksi AVX/AVX2 yang diperlukan TensorFlow.",
        "ERR_RUN_ON_COLAB": "Silakan: Jalankan pelatihan ini di Google Colab.",
    "CARD_SAVED": "Tersimpan \u2713",
    "CARD_TEMP": "Sementara",
    "CARD_SAVED_PARTIAL": "{s}/{t} Tersimpan",
    "BTN_DETAILS_COLLAPSED": "Detail \u25bc",
    "BTN_DETAILS_EXPANDED": "Detail \u25b2",
    "NO_METADATA": "Metadata model tidak tersedia (offline atau metadata.json hilang).",
    "DELETE_REGION": "Hapus Wilayah",
    "DELETE_SELECTED": "Hapus yang Dipilih",
    "DELETE_ALL": "Hapus Semua",
    "DELETE_JOB": "Hapus Semua",
    "VIEW_TILES": "Lihat Tile",
    "HIDE_TILES": "Sembunyikan",
    "REFRESH_MAP": "Muat Ulang Peta",
    "TASK_NAME_PLACEHOLDER": "Contoh: Klasifikasi Amazon Bawah 2025 (Lucas)",
    # M5 - Section titles
    "SECTION_MODEL": "1. Pilih Model (tunggal):",
    "SECTION_REGIONS": "2. Pilih Wilayah:",
    "SECTION_PERIODS": "3. Pilih Periode (Tahun / Tahun_Bulan):",
    "LOADING_CAMPAIGNS": "Memuat kampanye...",
    "CAMPAIGN_LABEL": "Kampanye:",
    "FILTER_LABEL": "Filter:",
    "FILTER_PLACEHOLDER": "cari berdasarkan model, wilayah, periode...",
    # M5 - Region diagnostic
    "ERR_CONFIG_ASSET_REGIONS": "CONFIG[asset_regions] tidak terdefinisi",
    "ERR_ASSET_NOT_FOUND": "Aset TIDAK DITEMUKAN di GEE:",
    "ERR_ASSET_EMPTY": "Aset ada tetapi KOSONG:",
    "ERR_ASSET_COLUMN": "Kolom <code>{prop}</code> tidak ditemukan",
    "ERR_GEE_ACCESS": "Kesalahan mengakses GEE:",
        "WARN_REGION_FALLBACK": "Tidak dapat mengisi wilayah secara otomatis",
        "ERR_RENDER_PENDING": "Error saat merender tugas tertunda:",
    "COLUMN_USED": "Kolom yang digunakan:",
    # M5 - Validation
    "ERR_NO_TASK_NAME": "Perhatian: Anda harus mengisi Nama Tugas.",
    "ERR_NO_SELECTION": "Perhatian: Pilih 1 Model dan setidaknya 1 Wilayah dan 1 Periode.",
    # M5 - Feedback messages
    "SAVING_TO_GCS": "Menyimpan ke GCS...",
    "MSG_NO_TASKS_ADDED": "Tidak ada tugas yang ditambahkan.",
    "MSG_TASKS_ADDED": "Berhasil: {n} tugas ditambahkan.",
    "FAIL_SAVE_LOCAL": "gagal menyimpan file lokal (lock sibuk)",
    "FAIL_GCS_UPLOAD": "{n} gagal di GCS",
    "MSG_GCS_OK": "{n} tersimpan di GCS.",
    "MSG_SKIPPED_QUEUED": "{n} dilewati (sudah dalam antrian).",
    "WARN_ALREADY_QUEUED": "Perhatian: {n} tugas sudah dalam antrian.",
    "MSG_LOADED": "{loaded} dimuat, {skipped} dilewati.",
    "TAREA_SAVED": "Tugas tersimpan di GCS untuk {m}.",
    "TAREA_DELETED": "Tugas dihapus dari GCS: {m}.",
    "WORKPLAN_DISCARDED": "Rencana kerja dibuang: {m} ({n} pekerjaan dihapus).",
    "QUEUE_CLEARED": "Antrian dikosongkan (GCS dihapus).",
    "SAVED_TO_GCS": "{n} pekerjaan tersimpan di GCS.",
    "ALREADY_SAVED": "{n} pekerjaan sudah tersimpan.",
    "TEMP_JOBS_REMOVED": "{n} pekerjaan sementara dihapus.",
    "NO_TEMP_JOBS": "Tidak ada pekerjaan sementara untuk dihapus.",
    "REMOVED_FROM_GCS": "{n} pekerjaan dihapus dari GCS.",
    "NO_TILES_GCS_ALT": "Tidak ada tile di GCS.",
    "TILES_DELETED": "{n} tile dihapus.",
    "SELECT_TILES_PROMPT": "Pilih tile untuk dihapus.",
    "SYNC_FROM_GCS": "{n} pekerjaan disinkronkan dari GCS.",
    # M5 - Tareas section
    "TAREAS_HEADER": "Tugas di GCS ({n} tersedia)",
    "TAREA_SUMMARY": "Model: {m} | Wilayah: {r} | Periode: {p}",
    # M5 - Scale bar
    "SCALE_BAR_WIDTH": "~{w} km lebar",
    # M5 - Pending debug
    "INFO_FILTER_COUNT": "{total} tertunda | {filtered} setelah filter",
    "GRID_COUNT": "({n} sel)",
    # M5 - Map legend
    "MAP_LEGEND_COUNTRY": "Negara",
    "MAP_LEGEND_REGIONS": "Wilayah",
    "MAP_LEGEND_GRID": "Grid cim-world",
    # M5 - Model inference errors
    "ERR_MODEL_NO_METADATA": "metadata.json tidak ditemukan di {path}",
    "ERR_MODEL_NO_BANDS_CONFIG": "Model tidak memiliki 'bands_config' di metadata.",
    "ERR_MODEL_NO_BAND_ORDER": "Model tidak memiliki 'band_order' di metadata. Latih ulang dengan M4 terbaru (commit 9b95392+).",
    "ERR_MODEL_INCONSISTENCY": "Inkonsistensi metadata model: {details}",
    # M5 - Jobs details panel
    "JOBS_REGISTERED": "Pekerjaan terdaftar",
    "JOBS_SUMMARY": "{total} pekerjaan, {regions} wilayah, {periods} periode",
    "JOB_REGION": "Wilayah",
    "JOB_PERIOD": "Periode",
    "JOB_STATUS": "Status",
    "JOB_TASK": "Tugas",
    # M5 - Model metadata panel
    "META_SENSORS": "Sensor",
    "META_PERIODICITY": "Periodisitas",
    "META_BANDS_INPUT": "Bands (input)",
    "META_ITERATIONS": "Iterasi",
    "META_LEARNING_RATE": "Learning rate",
    "META_CAMPAIGN": "Kampanye",
    # M5 Tabs
    "TAB_GUIDE": "Panduan",
    "TAB_REGISTER": "Daftar",
    "TAB_PENDING": "Tertunda",
    "TAB_PUBLISH": "Untuk Publikasi",
    "TAB_MAP": "Peta",
    "TAB_DONE": "Selesai",
    "GUIDE_M5_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #3498db; padding-bottom:5px;'>M5 - Klasifikasi Regional Skala Besar</h3>
        <p>Mengklasifikasikan beberapa wilayah (grid cim-world-1-250000) menggunakan model M4.</p>
        <h4>Alur:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_register}</b> — pilih model + wilayah + periode.</li>
            <li><b>{tab_pending}</b> — pantau klasifikasi tile per tile.</li>
            <li><b>{tab_map}</b> — gambaran umum pemrosesan langsung.</li>
            <li>Jalankan <code>run_m5_workplan()</code> di notebook untuk memproses.</li>
        </ol>
        <h4>Setelah Klasifikasi (M6):</h4>
        <p>Gunakan <b>M6</b> untuk membuat mosaik tile, menghasilkan statistik regional, dan unggah ke GEE.
        Buka UI M6 dan jalankan <code>run_m6_publish()</code> di sel terpisah.</p>
        <p>Tugas yang telah diklasifikasikan (COMPLETED) muncul secara otomatis di M6.</p>
    </div>""",
    "GUIDE_M1_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <div style='margin-bottom:15px;padding:10px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px;font-size:13px;'>
            <b>Catatan:</b> Saat ini hanya komposit bulanan untuk <b>Sentinel-2</b> dengan <b>minnbr</b> dan <b>minnbr_buffer</b> yang dirilis. Kombinasi sensor/periode/mozaik lainnya mungkin eksperimental.
        </div>
        <h3 style='color:#2c3e50; border-bottom:2px solid #e67e22; padding-bottom:5px;'>Modul 1 (M1) — Ekspor GEE Multi-Sensor</h3>
        <p>M1 mengekspor mozaik satelit bebas awan dari Google Earth Engine ke Google Cloud Storage (GCS) atau Aset GEE.
        Mendukung LANDSAT 5/7, LANDSAT 8/9, Sentinel-2, HLS, dan MODIS dengan koreksi radiometrik dan masking awan khusus per sensor.</p>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:15px;'>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Sensor dan Sumber</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>LANDSAT 5/7</b> — arsip historis (1984–)</li>
                    <li><b>LANDSAT 8/9</b> — OLI/TIRS terkini dengan pita termal</li>
                    <li><b>Sentinel-2</b> — resolusi 10m, kunjungan ulang 5 hari</li>
                    <li><b>HLS</b> — produk harmonisasi LANDSAT/Sentinel</li>
                    <li><b>MODIS</b> — cakupan global harian (250m–1km)</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Metode Mozaik</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li><b>minnbr</b> — awan paling sedikit berdasarkan peringkat NBR</li>
                    <li><b>minndvi</b> — awan paling sedikit berdasarkan peringkat NDVI</li>
                    <li><b>median</b> — komposit median per piksel</li>
                    <li><b>minnbr_buffer</b> — minnbr dengan masker buffer api INPE</li>
                </ul>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Alur Kerja</h4>
                <ol style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Pilih tab sensor (LANDSAT, S2, HLS, MODIS)</li>
                    <li>Pilih periode (bulanan / tahunan)</li>
                    <li>Pilih metode mozaik</li>
                    <li>Centang sel tanggal dan pita</li>
                    <li>Klik <b>Mulai Ekspor</b></li>
                </ol>
            </div>
            <div style='background:#fafafa;border:1px solid #eee;padding:15px;border-radius:6px;'>
                <h4 style='color:#e67e22;margin-top:0;'>Informasi Teknis</h4>
                <ul style='padding-left:18px;font-size:13px;line-height:1.6;'>
                    <li>Koreksi radiometrik per sensor</li>
                    <li>Masking awan via QA_PIXEL, Fmask, CS+</li>
                    <li>Masker buffer kebakaran INPE</li>
                    <li>Keluaran: fragmen GeoTIFF di GCS / ImageCollection di GEE</li>
                </ul>
            </div>
        </div>
        <div style='margin-top:15px;padding:10px;background:#fef3e2;border-left:4px solid #e67e22;border-radius:4px;font-size:13px;'>
            <b>Tips:</b> Gunakan <b>Sinkron Data</b> untuk memperbarui cache. Gunakan <b>Pilih Tertunda</b> untuk mencentang otomatis semua tanggal yang tersedia.
        </div>
    </div>""",
    "GUIDE_M2_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>Modul 2 (M2) — Perakitan Mozaik COG</h3>
        <p>M2 merakit fragmen GeoTIFF yang diekspor oleh M1 menjadi COG (Cloud-Optimized GeoTIFF) nasional lengkap menggunakan GDAL.</p>
        <h4>Alur:</h4>
        <ol style='line-height:1.6;'>
            <li><b>Periksa status</b> — OK (COG ada), READY (fragmen tersedia), MISS (tidak tersedia)</li>
            <li><b>Pilih sel</b> — centang pita yang ingin dirakit</li>
            <li>Klik <b>Mulai Perakitan</b></li>
            <li>GDAL mengunduh fragmen → membangun VRT → konversi ke COG (LZW) → unggah ke GCS</li>
        </ol>
        <h4>Persyaratan:</h4>
        <ul style='line-height:1.6;'>
            <li>GDAL terinstal (<code>gdalbuildvrt</code>, <code>gdal_translate</code>)</li>
            <li>M1 harus dijalankan terlebih dahulu untuk menghasilkan fragmen sumber</li>
        </ul>
        <div style='margin-top:15px;padding:10px;background:#e8f8ed;border-left:4px solid #27ae60;border-radius:4px;font-size:13px;'>
            <b>Tips:</b> COG disimpan di <code>.../COG/</code> dan dikonsumsi oleh M4 (pelatihan) dan M5 (klasifikasi).
        </div>
    </div>""",
    # Map / Grid
    "LIVE_PROCESSING": "Pemrosesan Langsung",
    "CURRENT_TILE": "Tile saat ini",
    "COMPLETED": "Selesai",
    "GRID_REGION": "Wilayah",
    "GRID_CELLS": "Sel cim-world",
    # Tiles / GCS
    "TILES": "tile",
    "MOSAIC": "mozaik",
    "STATS": "statistik",
    "SELECT_TILES_DELETE": "Pilih tile untuk dihapus.",
    "TILES_REMOVED": "tile dihapus.",
    # Task status
    "STATUS_QUEUED": "Dalam rencana",
    "STATUS_RUNNING": "Berjalan",
    "STATUS_COMPLETED": "Selesai",
    "STATUS_FINISHED": "Terselesaikan",
    "STATUS_ERROR": "Kesalahan",
    "STATUS_PUBLISHING": "Mempublikasi",
    "STATUS_PUBLISHED": "Dipublikasi",
    "STATUS_SKIPPED": "Dilewati",
    # M4 - Training
        "MODEL_TRAINER": "Pelatih Model",
        "M4_HEADER_TITLE": "M4 - Pelatih Model",
        "SELECTED_OK": "[OK] Dipilih",
        "TRAINING_PROGRESS": "Pelatihan:",
    "ITERATIONS": "Iterasi",
    "BATCH_SIZE": "Ukuran Batch",
    "LEARNING_RATE": "Laju Pembelajaran",
    "HIDDEN_LAYERS": "Lapisan Tersembunyi",
    "ACTIVATION": "Aktivasi",
    "DROPOUT": "Dropout",
    "OPTIMIZER": "Pengoptimal",
    "LOSS_FN": "Fungsi Kerugian",
    "METRICS": "Metrik",
    "SAMPLE_SELECTION": "Pemilihan Sampel",
    "EXTRACTION_MATRIX": "Matriks Ekstraksi",
    "MODEL_CONFIG": "Konfigurasi Model",
    "GCS_DEST": "Tujuan GCS",
    "USAGE_GUIDE": "Panduan Penggunaan",
    "GUIDE_M4_HTML": """<div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; background: #fdfdfd; color: #2c3e50; line-height: 1.6;">
        <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">M4 Model Trainer - Panduan Penggunaan</h1>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 20px;">
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #3498db; margin-top:0;">Struktur Platform</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>{usage_guide}:</b> Layar orientasi dan dokumentasi.</li>
                    <li><b>{new_training}:</b> Konfigurasi eksperimen baru, pemilihan sampel dan pita.</li>
                    <li><b>{trainings}:</b> Peringkat historis dengan metrik terperinci dan manajemen model.</li>
                    <li><b>Canvas:</b> Meja audit paralel untuk membandingkan beberapa model secara mendalam.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #9b59b6; margin-top:0;">Konsep Teknis</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>TensorFlow:</b> Mesin AI Google untuk perhitungan matematis masif.</li>
                    <li><b>DNN (Deep Neural Network):</b> Jaringan dalam yang meniru pembelajaran manusia.</li>
                    <li><b>Neuron:</b> Unit yang memproses sinyal dan mengaktifkan pola pembelajaran.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #e67e22; margin-top:0;">Hiperparameter (DNN)</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Layers:</b> Arsitektur jaringan. Lebih banyak lapisan menangkap detail lebih halus.</li>
                    <li><b>Learning Rate (LR):</b> Mengontrol seberapa cepat model menyesuaikan.</li>
                    <li><b>Epochs:</b> Siklus pelatihan lengkap pada kumpulan sampel.</li>
                    <li><b>Batch Size:</b> Blok data yang diproses sebelum setiap pembaruan.</li>
                </ul>
            </div>
            <div style="background: white; border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #27ae60; margin-top:0;">Kamus Kualitas</h3>
                <ul style="padding-left: 20px; font-size:13px;">
                    <li><b>Accuracy:</b> Persentase total kebenaran global.</li>
                    <li><b>Precision:</b> Ketepatan: Seberapa banyak api yang ditandai itu nyata? (Menghindari positif palsu).</li>
                    <li><b>Recall:</b> Cakupan: Seberapa banyak api nyata yang ditemukan? (Menghindari kelalaian).</li>
                    <li><b>F1-Score:</b> Rata-rata harmonis. Keseimbangan terbaik antara Precision dan Recall.</li>
                    <li><b>Nilai AI:</b> Audit otomatis yang menghukum keras kelalaian.</li>
                    <li><b>Nilai Manusia:</b> Evaluasi subjektif (1-5) pada Ruang Laten.</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 30px; padding: 15px; background: #e8f4fd; border-left: 5px solid #3498db; border-radius: 4px; font-size:14px;">
            <b>[Tips] Pro-Tip Auditor:</b> Gunakan <b>Canvas</b> untuk memuat model lama (benchmark) dan model baru Anda. Bandingkan apakah pemisahan kelas dalam t-SNE 3D telah membaik atau ada zona kebingungan baru.
        </div>
    </div>""",
    # M4 - Canvas
    "METADATA": "Metadata",
    "KPIS": "KPI",
    "CONFUSION": "Kebingungan",
    "HISTORY": "Riwayat",
    "PROB": "Prob",
    "PR_CURVE": "Kurva PR",
    "MANAGEMENT": "Manajemen",
    "RANKING": "Peringkat / Repositori",
    "SELECTED_CANVAS": "Dipilih di Canvas",
    "SEARCH_REPO": "Cari repositori...",
    "SEARCH_SAMPLES": "Cari sampel...",
    "SYNC_CATALOG": "Sinkron Katalog (GCS)",
    "APPLY_VISIBILITY": "Terapkan Visibilitas",
    # M1 / M2
    "SYNC_DATA": "Sinkron Data",
    "SELECT_PENDING": "Pilih Tertunda",
    "CLEAR_SELECTION": "Bersihkan Pilihan",
    "EXPORT_START": "Mulai Ekspor",
    "MOSAIC_START": "Mulai Perakitan",
    # M6
    "POST_CLASSIFICATION": "Pemroses Pasca-Klasifikasi",
    "USING_PRESET": "Menggunakan Konfigurasi Preset",
    "FILTER_START": "Mulai Penyaringan",
    "EXPORT_TASK_STARTED": "Tugas ekspor dimulai",
    "CONFIG_SUMMARY": "Ringkasan Konfigurasi",
    # M7
    "CURATOR": "Kurator",
    "CURATOR_DESC": "Publikasi Koleksi Pra-Resmi",
    "USING_VOTES_PRESET": "Menggunakan Suara Preset",
    "CURATION_START": "Mulai Kurasi",
    "EXPORT_ASSET_STARTED": "Ekspor GEE Asset dimulai",
    # Cache
    "CACHE_REMOVED": "Cache lokal dihapus",
    "CACHE_NOT_FOUND": "Cache tidak ditemukan",
    # Misc
    "LOADING_TILES": "Memuat tile...",
    "CONFIRM_DELETE": "Konfirmasi Penghapusan",
    "CONFIRM_DELETE_ALL": "Konfirmasi Hapus Semua",
    "CANCELED": "Dibatalkan.",
    "GLOBAL_OPTS_SET": "Opsi global dikonfigurasi",
    # Dropdown labels
    "DROP_MODEL": "Model:",
    "DROP_REGION": "Wilayah:",
    "DROP_YEAR": "Tahun:",
    "DROP_TASK": "Tugas:",
    "DROP_TASK_NAME": "Nama Tugas:",
    # Hyperparams
    "HP_EPOCHS": "Epoch",
    "HP_PATIENCE": "Kesabaran",
    "HP_TEST_SPLIT": "Split Uji",
    "HP_BALANCE": "Keseimbangan",
    "HP_AUGMENT": "Augmentasi Data",
    "HP_ACTIVATION": "Fungsi Aktivasi",
    "HP_OPTIMIZER": "Pengoptimal",
    "HP_LOSS": "Fungsi Kerugian",
    # Extraction
    "EXTRACTION_TITLE": "Matriks Ekstraksi (Multisensor GCS)",
    "SAMPLE_SELECT": "Pemilihan Sampel",
    "AVAILABLE": "Tersedia",
    "SELECTED": "Dipilih",
    "ADD": "Tambah",
    "REMOVE": "Hapus",
    "CAMPAIGN": "Kampanye",
    "SENSOR": "Sensor",
    "BANDS": "Band",
    "STATUS_OK": "OK",
    "STATUS_RUN": "Jal.",
    "STATUS_MISS": "Hil.",
    # Training
    "TRAINING_ID": "ID Pelatihan:",
    "SHORTNAME": "Nama Singkat:",
    "COMMENTS": "Komentar...",
    "START_TRAINING": "Mulai Pelatihan",
    "CANVAS_TITLE": "Pusat Pelatihan dan Audit",
    "CANVAS_EMPTY": "Canvas kosong",
    "CANVAS_HINT": "Jelajahi dan pilih model di panel samping untuk melihatnya di sini.",
    "LOSS_ACC": "Loss / Akurasi",
    "VIZ_METADATA": "Metadata",
    "VIZ_KPIS": "KPI",
    "VIZ_CONFUSION": "Kebingungan",
    "VIZ_HISTORY": "Riwayat",
    "VIZ_PROB": "Prob",
    "VIZ_PR_CURVE": "Kurva PR",
    "VIZ_MANAGEMENT": "Manajemen",
    # Repository
    "REPO_TITLE": "Peringkat / Repositori",
    "REPO_SEARCH": "Cari repositori...",
    "REPO_SYNC": "Sinkron GCS",
    "REPO_ALL": "Semua",
    "REPO_CLEAR": "Bersihkan",
    "REPO_SYNC_CATALOG": "Sinkron Katalog (GCS)",
    "REPO_SCANNING": "Memindai GCS... Harap tunggu.",
    "REPO_SCAN_DONE": "Katalog tersinkronisasi!",
    # Tab titles
    "SORT_BY": "Urutkan:",
    "NEW_TRAINING": "Pelatihan Baru",
    "TRAININGS": "Pelatihan",
    # M1 - Export
    "EXPORT_TITLE": "Ekspor Koleksi",
    "EXPORT_SYNC": "Sinkron Data",
    "EXPORT_SELECT": "Pilih Tertunda",
    "EXPORT_CLEAR": "Bersihkan Pilihan",
    "EXPORT_START_BTN": "Mulai Ekspor",
    "EXPORT_SENT": "tugas terkirim.",
    "EXPORT_NONE_SEL": "Tidak ada opsi dipilih untuk ekspor.",
    # M2 - Mosaic
    "MOSAIC_TITLE": "Perakitan Mozaik",
    "MOSAIC_SYNC": "Sinkron Data",
    "MOSAIC_SELECT": "Pilih Tertunda",
    "MOSAIC_CLEAR": "Bersihkan Pilihan",
    "MOSAIC_START_BTN": "Mulai Perakitan",
    "MOSAIC_DONE": "Perakitan berhasil diselesaikan",
    "M1_HEADER_TITLE": "M1 - Operator",
    "M1_HEADER_SUBTITLE": "Antarmuka multi-sensor untuk ekspor",
    "M1_TITLE_FULL": "M1 - Operator Mozaik",
    "M1_DESCRIPTION": "Antarmuka multi-sensor untuk mengirim komposisi (Assets/GCS) ke cloud.",
    "LABEL_PROJECT": "Proyek",
    "BTN_SELECT_ROW": "[S]",
    "COL_TYPE": "Tipe",
    "M2_HEADER_TITLE": "M2 - Perakit",
    "M2_HEADER_SUBTITLE": "Antarmuka perakitan mozaik COG nasional",
    "M2_CONSTRUCTOR_TITLE": "M2 - Perakit Mozaik (COG)",
    "M2_CONSTRUCTOR_DESC": "Antarmuka untuk mengonversi fragmen GCS menjadi mozaik COG nasional.",
    "STATUS_READY": "SIAP",
    # M0 - Startup
    "START_COUNTRY": "Negara",
    "START_BUCKET": "Bucket",
    "START_VERSION": "Versi",
    "START_SENSOR": "Sensor",
    "START_PERIODICITY": "Periodisitas",
    "START_CAMPAIGN": "Kampanye",
    "START_FOUND": "ditemukan",
    "START_NOT_FOUND": "tidak ditemukan",
    "START_GDAL_OK": "GDAL ditemukan dan ditambahkan ke PATH",
    "START_GDAL_MISSING": "Peringatan: Utilitas GDAL tidak ditemukan",
    "START_COLAB_HINT": "Di Google Colab, jalankan: !apt-get install -y gdal-bin",
    "START_WIN_HINT": "Di Windows, pastikan GDAL ada di PATH Anda",
    # M6
    "M6_TITLE": "Pemroses Pasca-Klasifikasi",
    "M6_PRESET": "Menggunakan Konfigurasi Preset",
    "M6_DOWNLOAD": "Mengunduh fragmen",
    "M6_VRT": "Membangun VRT",
    "M6_COG": "Mengonversi ke COG",
    "M6_UPLOAD": "Diunggah",
    "M6_START": "Memulai Penyaringan",
    "M6_EXPORT_OK": "Tugas ekspor dimulai",
    "M6_SUMMARY": "Ringkasan Konfigurasi",
    "TAB_ANALYTICS": "Analitik",
    "TAB_M6_COVERAGE": "Cakupan",
    "GUIDE_M6_HTML": """<div style='padding:20px; font-family:sans-serif;'>
        <h3 style='color:#2c3e50; border-bottom:2px solid #27ae60; padding-bottom:5px;'>M6 - Mozaik, Statistik & Publikasi</h3>
        <p>Menerbitkan tile terklasifikasi dari M5: membuat mosaik regional, menghitung statistik area terbakar, dan mengunggah ke Google Earth Engine.</p>
        <h4>Alur:</h4>
        <ol style='line-height:1.6;'>
            <li><b>{tab_publish}</b> — grup tile terklasifikasi menunggu mozaik.</li>
            <li><b>{tab_done}</b> — wilayah diterbitkan dengan mozaik dan statistik.</li>
            <li><b>{tab_analytics}</b> — tabel statistik gabungan dengan unduhan.</li>
            <li><b>{tab_coverage}</b> — peta cakupan: diterbitkan vs tertunda.</li>
            <li>Jalankan <code>run_m6_publish()</code> di notebook untuk memproses.</li>
        </ol>
    </div>""",
    "DOWNLOAD_TABLE": "Unduh Tabel",
    "ANALYTICS_FILTER_MODEL": "Model",
    "ANALYTICS_FILTER_REGION": "Wilayah",
    "ANALYTICS_FILTER_PERIOD": "Periode",
    "REFRESH_M6": "Muat Ulang M6",
    "M6_HEADER_TITLE": "M6 - Mozaik, Statistik & Publikasi",
    "M6_LABEL_PERIOD": "Periode:",
    "M6_GROUPS_PENDING": "{n} grup menunggu mozaik",
    "M6_MOSAIC_OK": "mozaik OK",
    "M6_DOWNLOAD_MOSAIC": "OK \u2197",
    "M6_PUBLISHED_GROUPS": "{n} grup dipublikasikan",
    "M6_BADGE_MOSAIC": "Mosaik",
    "M6_BADGE_STATS": "Statistik",
    "M6_BADGE_GEE": "GEE",
    "M6_COL_PCT": "%",
    "M6_COL_HA": "ha",
    "M6_COL_TILES": "Ubin",
    "M6_LEGEND_PUBLISHED": "Dipublikasikan",
    "M6_LEGEND_PARTIAL": "Parsial",
    "M6_LEGEND_CLASSIFIED_ONLY": "Hanya diklasifikasi",
    "M6_LEGEND_NO_DATA": "Tidak ada data",
    "M6_N_RECORDS": "{n} catatan",
    "M6_DOWNLOAD_LINK": "Unduh {fname}",
    "M6_NO_STATS": "Tidak ada statistik terkonsolidasi. Jalankan M6 publish terlebih dahulu.",
    "M6_NO_MATCHING": "Tidak ada catatan yang cocok.",
    "M6_NO_CLASSIFIED_GROUPS": "Tidak ada grup terklasifikasi ditemukan.",
    # M7
    "M7_TITLE": "Kurator",
    "M7_DESC": "Publikasi Koleksi Pra-Resmi",
    "M7_PRESET": "Menggunakan Suara Preset",
    "M7_START": "Memulai Kurasi",
    "M7_EXPORT_OK": "Ekspor GEE Asset dimulai untuk",
    "M7_SUMMARY": "Ringkasan Konfigurasi",
    # M3
    "M3_TITLE": "M3 - Koleksi Sampel (Gateway Toolkit GEE)",
    "M3_SOURCE": "Kode sumber (GitHub)",
    "M3_EDITOR": "Akses langsung (Editor GEE)",
    "M3_DOCS": "Dokumentasi dan panduan penggunaan",
    "M3_SLIDES": "Slide tutorial (Google Slides)",
    "M3_INTRO_LINE1": "Pengumpulan sampel dilakukan secara eksklusif",
    "M3_INTRO_LINE2": "melalui antarmuka JavaScript di Google Earth Engine.",
    # M4 - Analytics / KPIs
    "ACCURACY": "Akurasi",
    "PRECISION": "Presisi",
    "RECALL": "Recall",
    "F1_SCORE": "F1-Score",
    "AI_NOTE": "Catatan AI",
    "HUMAN_NOTE": "Catatan Manusia",
    "NO_METRICS": "Tidak ada metrik",
    "CONFUSION_MATRIX": "Matriks Kebingungan (%)",
    "HISTORICAL_EVOLUTION": "Evolusi Historis",
    "PROB_DISTRIBUTION": "Distribusi Probabilitas",
    "CONFIDENCE": "Keyakinan",
    "LATENT_PROJ_2D": "Proyeksi Laten 2D",
    "COST_LOSS": "Biaya (Loss)",
    "CLASSIC_METRICS": "Metrik Klasik dan Proyeksi Statis",
    "INTERACTIVE_LATENT": "Ruang Laten Interaktif",
    "PCA_3D": "PCA 3D",
    "TSNE_3D": "t-SNE 3D",
    "PCA_3D_INTERACTIVE": "PCA 3D Interaktif",
    "TSNE_3D_INTERACTIVE": "t-SNE 3D Interaktif",
    "RETRAIN": "Latih Ulang",
    # M4 - Fire class labels
    "FIRE": "Api",
    "NO_FIRE": "Bukan-api",
    "FIRE_CLASS": "Api",
    "NO_FIRE_CLASS": "Bukan-api",
    # M4 - UI Labels
    "BASIC_STATS": "Statistik Dasar",
    "PCA_LATENT": "Ruang Laten PCA",
    "TSNE_LATENT": "Ruang Laten t-SNE",
    "VIZ_OPTIONS": "Opsi Visualisasi",
    "VIZ_PCA2D": "PCA 2D",
    "VIZ_PCA3D_STATIC": "PCA 3D (Statis)",
    "VIZ_PCA3D_INTERACTIVE": "PCA 3D (Interaktif)",
    "VIZ_TSNE3D_STATIC": "t-SNE 3D (Statis)",
    "VIZ_TSNE3D_INTERACTIVE": "t-SNE 3D (Interaktif)",
    "HYPERPARAMS_SECTION": "Hiperparameter (DNN)",
    "LIVE_TRAINING": "Pelatihan Langsung",
    "TRAINING_IN_PROGRESS": "Pelatihan berlangsung...",
    "LIVE_TSNE_AUDIT": "Audit t-SNE Langsung (Ruang Laten)",
    "ADD_TO_CANVAS": "Tambah ke Canvas",
    "REMOVE_FROM_CANVAS": "Hapus dari Canvas",
    "BTN_CLOSE": "X",
    "RELOAD_SAMPLES": "Muat ulang daftar sampel dari GCS",
    # M4 - Analytics card labels
    "LAYERS_LABEL": "Lapisan:",
    "LR_ABBR": "LR:",
    "SAMPLES_LABEL": "Sampel:",
    "NO_COMMENTS": "Tidak ada komentar.",
    "HIDE_ADVANCED": "Sembunyikan parameter lanjutan",
    "SHOW_ALL_PARAMS": "Tampilkan semua parameter",
    "TSNE_AXIS_1": "t-SNE 1",
    "TSNE_AXIS_2": "t-SNE 2",
    "TSNE_AXIS_3": "t-SNE 3",
    "ACC_ABBR": "Aku",
    "F1_ABBR": "F1",
}

SUPPORTED_LOCALES = {
    "en": None,        # English is the default (class attributes)
    "es": STRINGS_ES,
    "pt": STRINGS_PT,
    "fr": STRINGS_FR,
    "id": STRINGS_ID,
}

LOCALE_LABELS = {
    "en": "English",
    "es": "Español",
    "pt": "Português",
    "fr": "Français",
    "id": "Bahasa Indonesia",
}


# ── Locale switching ─────────────────────────────────────────

def _load_locale(cls, locale="en"):
    """Swap all class attributes to the given locale.

    - English ('en') → restores the original class defaults.
    - Others ('es', 'pt', 'fr', 'id') → applies the corresponding STRINGS_XX dict.
    """
    strings = SUPPORTED_LOCALES.get(locale)
    if strings is None:
        if locale == "en":
            return  # already English defaults
        raise ValueError(f"Locale '{locale}' is not supported. Supported: {list(SUPPORTED_LOCALES)}")
    for k, v in strings.items():
        if hasattr(cls, k):
            setattr(cls, k, v)


# Install the method on the class
L.load_locale = classmethod(_load_locale)
