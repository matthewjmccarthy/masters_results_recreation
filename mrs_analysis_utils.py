import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import datetime
#from sklearn.preprocessing import normalize
#from scipy.stats import pearsonr, linregress

# --- File Handling Utilities ---

def xml_to_svu(tissue_type: str, parameters: dict, points: str) -> ET.ElementTree:
    """Generate SVU XML structure for MATLAB cNMF."""
    new_root = ET.Element('DATASET', CreatedBy='jMRUI2XML',
                          Date=f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}',
                          Version='1.0')
    grid  = ET.SubElement(new_root, 'Grid')
    voxel = ET.SubElement(grid, 'Voxel', FirstPPM=parameters['FirstPPM'],
                          LastPPM=parameters['LastPPM'], PointsNumber=parameters['PointsNumber'],
                          Xaxis='1', Yaxis='1', Zaxis='1', caseID=parameters['caseID'])
    tissue         = ET.SubElement(voxel, 'Tissue', Type=tissue_type)
    points_element = ET.SubElement(voxel, 'Points')
    points_element.text = points
    return new_root
    
def get_PPM(point, NPoint, MaxPPM, MinPPM) -> int:
    """Convert PPM value to point index in spectrum."""
    NPoint, MinPPM, MaxPPM = float(NPoint), float(MinPPM), float(MaxPPM)
    delta = abs(MaxPPM - MinPPM) / (NPoint - 1)
    idx_float = (MinPPM - point) / delta
    return int(round(idx_float))

# --- XML Reading Utilities ---

def read_xml(input_dir: str, output_dir: str, exp: list, rename=[], rename_to='') -> None:
    """
    Read XML spectra, optionally rename tissue types, save to output folder.
    Preserves original intensities (no normalization).
    """
    os.makedirs(output_dir, exist_ok=True)

    tree = ET.parse(os.path.join(input_dir))
    root = tree.getroot()

    for case in root.findall('Case'):
        case_id = case.get('ID')
        tissue = case.find('Tissue')
        tissue_type = tissue.get('Type')

        if rename and tissue_type in rename:
            tissue.set('Type', rename_to)
            tissue_type = rename_to

        if tissue_type not in exp:
            continue

        spectrum = case.find('Spectrum')
        parameters = {
            'FirstPPM': spectrum.find('Parameters').get('FirstPPM'),
            'LastPPM': spectrum.find('Parameters').get('LastPPM'),
            'PointsNumber': spectrum.find('Parameters').get('PointsNumber'),
            'caseID': case_id
        }
        points = np.array(spectrum.find('Points').text.strip().split(), dtype=np.float64)
        points_str = ' '.join(map(str, points))  # keep raw intensities

        new_tree = xml_to_svu(tissue_type, parameters, points_str)
        ET.ElementTree(new_tree).write(os.path.join(output_dir, f'{case_id}.xml'),
                                        encoding='utf-8', xml_declaration=True)

    print("XML processing complete.")

# --- Data Extraction Utilities ---

def get_mean_data(Data_filtered, labels, xaxis, exp, results_dir=None):
    """
    Compute mean spectra per label/class.
    Returns dict: {label: mean_spectrum}
    """
    mean_dict = {}
    unique_labels = sorted(set(labels))

    for lab in unique_labels:
        mean_dict[lab] = Data_filtered[labels == lab].mean(axis=0)

    return mean_dict


def extract_filtered_data(input_dir: str, ppm_range: list, exp: list):
    """Extract spectra, labels, case_ids for given ppm range."""
    Data, labels, case_ids = [], [], []

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(input_dir, file))
        root = tree.getroot()

        points = np.array(root.find('.//Points').text.strip().split(), dtype=np.float64)
        label  = root.find('.//Tissue').get('Type')

        voxel = root.find('.//Voxel')
        NPoint = int(voxel.get('PointsNumber'))
        ppm_first, ppm_last = float(voxel.get('FirstPPM')), float(voxel.get('LastPPM'))

        if label not in exp:
            continue

        # ---- Case ID extraction (preferred: from Voxel attribute) ----
        case_id = voxel.get('caseID')
        if case_id is None:
            # fallback: filename without extension
            case_id = os.path.splitext(file)[0]

        min_idx = get_PPM(ppm_range[0], NPoint, ppm_last, ppm_first)
        max_idx = get_PPM(ppm_range[1], NPoint, ppm_last, ppm_first)

        Data.append(points[max_idx-1:min_idx])
        labels.append(label)
        case_ids.append(case_id)

    Data = np.array(Data)
    labels = np.array(labels).astype(str)
    case_ids = np.array(case_ids).astype(str)

    xaxis = np.flip(np.linspace(ppm_range[0], ppm_range[1], Data.shape[1], endpoint=True))
    return Data, labels, case_ids, xaxis

# --- Source + Weight Utilities ---

def load_sources(input_dir: str, results_dir: str, ppm_range: list,
                 good_idx: int, exp: list):
    Data_filtered, labels, case_ids, xaxis = extract_filtered_data(
        input_dir, ppm_range, exp
    )

    raw_signals = Data_filtered.copy()
    mean_data = get_mean_data(Data_filtered, labels, xaxis, exp)

    sources = []
    for file in sorted(os.listdir(results_dir)):
        if file.endswith('.xlsx') and file.startswith(f'Iteration{good_idx}_'):
            arr = pd.read_excel(os.path.join(results_dir, file), header=None).values.flatten()
            sources.append(arr)

    sources = np.array(sources)

    return sources, mean_data, labels, case_ids, xaxis, raw_signals

def load_weights(results_dir: str, good_idx: int):
    """Load H matrix from Excel files."""
    H = []
    for file in sorted(os.listdir(results_dir)):
        if file.endswith(f'n{good_idx}.xlsx') and file.startswith('W'):
            H.extend(pd.read_excel(os.path.join(results_dir, file), header=None).values)
    return np.array(H)


# --- Reconstruction Utilities ---

def reconstruct_spectra(H: np.ndarray, sources: np.ndarray, raw_data: np.ndarray = None):
    """Reconstruct spectra from weights and sources. Optionally rescale to raw_data max."""
    recon = np.dot(H, sources)
    if raw_data is not None:
        recon *= np.max(raw_data) / np.max(recon)
    return recon