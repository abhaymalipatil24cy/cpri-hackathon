from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
OUTPUTS_DIR = ROOT_DIR / 'outputs'

TRAIN_DATA_PATH = DATA_DIR / 'training_data.csv'
TEST_DATA_PATH = DATA_DIR / 'test_data.csv'

RANDOM_SEED = 42
SENTINEL_CANDIDATES = [18.0, 25.0, 55.0]
REGIME_CLUSTER_CANDIDATES = [4, 5, 6]
DEFAULT_K_CLUSTERS = 4
Z_SCORE_THRESHOLD = 3.0

OPERATING_VARIABLES = ['Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min']
SENSOR_VARIABLES = ['Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4']
INPUT_FEATURES = OPERATING_VARIABLES + SENSOR_VARIABLES
TARGET_REFERENCE = 'Reference_Parameter'
TARGET_VALIDITY = 'Validity_Label'
