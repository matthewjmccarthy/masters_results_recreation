import os
import numpy as np
import pandas as pd # type: ignore
import datetime
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from sklearn.preprocessing import normalize # type: ignore
from scipy.stats import pearsonr, linregress

def xml_to_svu(tissue_type: str, parameters: dict, points: str) -> ET.ElementTree:
    '''
    Converts a given xml tree into the SVU format needed for MATLAB cNMF function
    '''
    new_root = ET.Element('DATASET', CreatedBy='jMRUI2XML', Date='{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now()), Version='1.0')
    grid = ET.SubElement(new_root, 'Grid')
    voxel = ET.SubElement(grid, 'Voxel', FirstPPM=parameters['FirstPPM'], LastPPM=parameters['LastPPM'],
                            PointsNumber=parameters['PointsNumber'], Xaxis='1', Yaxis='1', Zaxis='1', 
                            caseID=parameters['caseID'])
    tissue         = ET.SubElement(voxel, 'Tissue', Type=tissue_type)
    points_element = ET.SubElement(voxel, 'Points')
    points_element.text = points

    return new_root

def get_mean_data(data, labels, xaxis, exp, output_dir) -> dict:
    mean_data = {label: data[labels == label].mean(axis=0) for label in exp}
    
    return mean_data

def read_xml_file(xml_path: str, output_dir: str, exp: list, rename=[], rename_to='') -> None:
    '''
    Read in single source file and split into multiple case files in a selected
    folder
    '''
    tree = ET.parse(xml_path) #
    root = tree.getroot()

    for case in root.findall('Case'):
        case_id = case.get('ID')
        tissue = case.find('Tissue')
        tissue_type = tissue.get('Type')

        if rename and tissue_type in rename:
            tissue.set('Type', rename_to)
            tissue_type = rename_to

        if tissue_type not in exp: #
            continue

        spectrum = case.find('Spectrum')
        parameters = {
            'FirstPPM': spectrum.find('Parameters').get('FirstPPM'),
            'LastPPM': spectrum.find('Parameters').get('LastPPM'),
            'PointsNumber': spectrum.find('Parameters').get('PointsNumber'),
            'caseID': case_id,
        }
        points = spectrum.find('Points').text.strip()

        point_array = np.array(points.split(), dtype=np.float64) #.reshape(1, -1)
        #normed_points = normalize(point_array, axis=1, norm='l2').flatten()
        points = ' '.join(map(str, point_array))
        
        # Create new XML structure
        new_tree = xml_to_svu(tissue_type, parameters, points)

        # Write to a new XML file
        output_file = f'{output_dir}/{case_id}.xml' 
        new_tree = ET.ElementTree(new_tree)
        new_tree.write(output_file, encoding='utf-8', xml_declaration=True)
        #print(new_tree)

    print('Processing complete.')

def get_PPM(point, NPoint, MaxPPM, MinPPM) -> int:
    '''
    Converts a point in PPM to its index in a given range
    '''
    NPoint, MinPPM, MaxPPM = float(NPoint), float(MinPPM), float(MaxPPM)
    delta = abs(MaxPPM - MinPPM) / (NPoint - 1)
    idx_float = (MinPPM - point) / delta
    idx = int(round(idx_float))

    return idx


def extract_filtered_data_and_labels(input_dir: str, ppm_range: list, exp: list):
    Data_filtered_m, labels_m = [], []

    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.xml') and file.startswith('I'):
            tree = ET.parse(os.path.join(input_dir, file))
            root = tree.getroot()
            points = list(map(float, root.find('.//Points').text.strip().split()))
            label = root.find('.//Tissue').get('Type')
            voxel = root.find('.//Voxel')
            ppm_first = voxel.get('FirstPPM')
            ppm_last = voxel.get('LastPPM')
            num_points = voxel.get('PointsNumber')

            labels_m.append(label)

            if ppm_range[0] < float(ppm_first):
                ppm_range[0] = ppm_first
            if ppm_range[1] > float(ppm_last):
                ppm_range[1] = ppm_last

            min_point = get_PPM(ppm_range[0], num_points, ppm_last, ppm_first)
            max_point = get_PPM(ppm_range[1], num_points, ppm_last, ppm_first)
            points_filtered = points[max_point - 1:min_point]

            Data_filtered_m.append(points_filtered)

    Data_filtered_m = np.array(Data_filtered_m)
    labels_m = np.array(labels_m)

    xaxis = np.flip(np.linspace(ppm_range[0], ppm_range[1], len(points_filtered), endpoint=True))
    return Data_filtered_m, labels_m, xaxis

def sources_from_xml(input_dir: str, results_dir: str, ppm_range: list, good_idx: int, exp: list, plot_setting: bool=False):
    Data_filtered_m, labels_m, xaxis = extract_filtered_data_and_labels(input_dir, ppm_range, exp)
    mean_Data_filtered_m = get_mean_data(Data_filtered_m, labels_m, xaxis, exp, results_dir)

    Sources_m = []
    for file in sorted(os.listdir(results_dir)):
        if file.endswith('.xlsx') and file.startswith(f'Iteration{good_idx}_'):
            points = pd.read_excel(os.path.join(results_dir, file), header=None)
            Sources_m.append(points.values.flatten())
    Sources_m = np.array(Sources_m)

    n_sources = len(Sources_m)
    n_means = len(mean_Data_filtered_m)
    print(n_sources, n_means)
    mean_keys = list(mean_Data_filtered_m.keys())

    if plot_setting==True:
        fig, axs = plt.subplots(n_means, n_sources, figsize=(4 * n_sources, 4 * n_means), squeeze=False)
        for row_idx, mean_label in enumerate(mean_keys):
            mean_vals = mean_Data_filtered_m[mean_label]
            for col_idx, source_vals in enumerate(Sources_m):
                ax = axs[row_idx, col_idx]
                ax.plot(xaxis, source_vals, label='Source')
                ax.plot(xaxis, mean_vals, label=f'Mean {mean_label.upper()}')
                ax.invert_xaxis()
                ax.grid(True)
                ax.set_title(f'Source {col_idx + 1} vs Mean {mean_label.upper()}')
                ax.legend()

        plt.tight_layout()
        plt.savefig(f'{results_dir}/source_vs_mean_grid_iteration{good_idx}.png')
        plt.show()

    return Sources_m, mean_Data_filtered_m, labels_m, xaxis, Data_filtered_m


def get_corr_cluster(source_data, mean_data, output_dir, exp):
    correlation_centroid_results = []

    fig, axs = plt.subplots(len(mean_data), source_data.shape[0], figsize=(15,15))

    for i, mean_labels in enumerate(mean_data):
        mean_vals = mean_data[mean_labels]
        for j, source_vals in enumerate(source_data):
            correlation_cloud = np.column_stack((mean_vals, source_vals))

            centroid_x = np.mean(correlation_cloud[:, 0])
            centroid_y = np.mean(correlation_cloud[:, 1])

            centroid_corr = np.sqrt(centroid_x**2 + centroid_y**2)

            cloud_corr, _ = pearsonr(correlation_cloud[:, 0], correlation_cloud[:, 1])

            slope, intercept, _, _, _ = linregress(correlation_cloud[:, 0], correlation_cloud[:, 1])
            regression_line_x = np.linspace(min(correlation_cloud[:, 0]), max(correlation_cloud[:, 0]), 100)
            regression_line_y = slope * regression_line_x + intercept

            ax = axs[i, j]
            ax.scatter(correlation_cloud[:, 0], correlation_cloud[:, 1], alpha=0.6, label="Correlation points")
            ax.scatter(centroid_x, centroid_y, color='red', marker='x', s=100, label="Centroid")
            ax.plot(regression_line_x, regression_line_y, color='blue', linestyle='-', label=f"Fit line: corr={cloud_corr:.3f}")
            ax.plot(regression_line_x, regression_line_x, color='black', linestyle='--', alpha=0.6)
            ax.axhline(0, color='grey', linestyle='--')
            ax.axvline(0, color='grey', linestyle='--')
            ax.set_xlabel(f"Source {j} Intensity", fontsize=20)
            ax.set_ylabel(f"Mean Spectrum Intensity of {exp[i]}", fontsize=20)
            ax.set_title(f"{exp[i]} vs Source {j}", fontsize=25, pad=15)
            ax.legend(fontsize=16)
            ax.grid(True)

            correlation_centroid_results.append({
                'Source_Label': exp[i],
                'Class': exp[j],
                'Corr': cloud_corr,
                'Centroid_Correlation': centroid_corr,
            })

    plt.tight_layout()
    plt.savefig(f"{output_dir}/corr_multiplot.png")
    plt.show()


    return pd.DataFrame(correlation_centroid_results)

def read_xml_file_anon(xml_path: str, output_dir: str, exp: list):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for case in root.findall('Case'):
        case_id = case.get('ID')
        tissue = case.find('Tissue')
        tissue_type = tissue.get('Type')

        if tissue_type not in exp: #
            continue

        spectrum = case.find('Spectrum')
        parameters = {
            'FirstPPM': spectrum.find('Parameters').get('FirstPPM'),
            'LastPPM': spectrum.find('Parameters').get('LastPPM'),
            'PointsNumber': spectrum.find('Parameters').get('PointsNumber'),
            'caseID' : case_id,
        }
        points = spectrum.find('Points').text.strip()

        point_array = np.array(points.split(), dtype=np.float64).reshape(1, -1)
        normed_points = normalize(point_array, axis=1, norm='l2').flatten()
        points = ' '.join(map(str, normed_points))
        
        # Create new XML structure
        new_tree = xml_to_svu('', parameters, points)

        # Write to a new XML file
        output_file = f'{output_dir}/{case_id}.xml' 
        new_tree = ET.ElementTree(new_tree)
        new_tree.write(output_file, encoding='utf-8', xml_declaration=True)
        #print(new_tree)

    print('Processing complete.')


def weights_from_xml(input_dir: str, results_dir: str, ppm_range: list, good_idx: int, exp: list):
    H = []
    for file in sorted(os.listdir(f'{results_dir}')):
        if file.endswith(f'n{good_idx}.xlsx') & file.startswith('W'):
            df = pd.read_excel(f'{results_dir}/{file}', header=None)
            H.extend(df.values)

    H = np.array(H)

    return H

def reconstruction_from_xml(input_dir: str, good_idx: int):
    reconstructed = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.xlsx') & file.startswith('F'):
            df = pd.read_excel(f'{input_dir}/{file}', header=None)
            reconstructed.extend(df.values)
            
    return np.array(reconstructed)[1:, 4:].astype(float)

def reconstruct_from_xml(input_dir: str, results_dir: str, ppm_range: list, good_idx: int, exp: list):
    return

