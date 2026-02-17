# config.py
# ================= Path Configuration =================
# Output directory for plots
OUTPUT_DIR = './results/plots'

# ================= Mode A: Method Comparison =================
# Add your CSV files here for comparison
COMPARISON_DATA_SOURCES = [
    {"file": './pred_csv/rcnn-matching_results_status.csv', "label": 'Oriented_RCNN'},
    {"file": './pred_csv/psc-matching_results_status.csv', "label": 'PSC'},
    {"file": './pred_csv/roi_transformer-matching_results_status.csv', "label": 'RoI_Transformer'},
    {"file": './pred_csv/reppoints-matching_results_status.csv', "label": 'RepPoints'},
    {"file": './pred_csv/kfiou-matching_results_status.csv', "label": 'KFIoU'},
    {"file": './pred_csv/h2rbox_v2-matching_results_status.csv', "label": 'H2RBox_v2'}
    # {"file": 'your_results.csv', "label": 'your_method'},
    # Add more methods here...
]

PLOT_CONFIGS = [
    {
        "name": "0-3000", 
        "min": 0, 
        "max": 3000, 
        "bin": 50, 
        "step": 200, 
        "highlights": [170, 610] 
    },
]

# ================= Mode B: Data Analysis (Analyzer) =================
ANALYSIS_FILE_PATH = './pred_csv/iou_statistics.csv'

# Parameters
KNEEDLE_GLOBAL_LIMIT = 5000
OUTLIER_SIGMA = 3         