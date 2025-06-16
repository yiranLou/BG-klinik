"""
C3D file conversion module.
Handles conversion between C3D files and OpenSim formats.
"""

import os
import numpy as np
import pyc3dserver as c3d
from scipy.spatial.transform import Rotation as R

# Relative path to project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Output directory for processed data
PROCESSED_DIR = r"/newPipeline/data/processed"

def ensure_dir_exists(directory):
    """
    Ensure directory exists, create if it doesn't exist.

    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def c3dtotrc(src_c3d, output_dir=None):
    """
    Convert C3D file to TRC and MOT formats.

    Args:
        src_c3d (str): Source C3D file path
        output_dir (str, optional): Output directory, defaults to processed data directory

    Returns:
        tuple: (trc file path, mot file path)
    """
    try:
        print(f"[DEBUG] c3dtotrc called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        if output_dir is None:
            output_dir = PROCESSED_DIR

        print(f"[DEBUG] Using output directory: {output_dir}")
        ensure_dir_exists(output_dir)

        src_file_name = os.path.basename(src_c3d)
        file_name_without_ext = os.path.splitext(src_file_name)[0]

        trc_path = os.path.join(output_dir, file_name_without_ext + ".trc")
        mot_path = os.path.join(output_dir, file_name_without_ext + ".mot")

        print(f"[DEBUG] Will write TRC to: {trc_path}")
        print(f"[DEBUG] Will write MOT to: {mot_path}")

        # Create C3Dserver interface
        itf = c3d.c3dserver()
        # Initialize logging
        c3d.init_logger(logger_lvl='DEBUG', c_hdlr_lvl='DEBUG', f_hdlr_lvl='DEBUG', f_hdlr_f_path=None)
        # Open C3D file
        ret = c3d.open_c3d(itf, src_c3d)

        # Define original coordinate axes and target coordinate axes
        axes_src = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axes_tgt = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

        # Calculate rotation matrix
        rot_ret = R.align_vectors(a=axes_src, b=axes_tgt)
        rot_obj = rot_ret[0]
        trf = rot_obj.as_matrix()

        # Export TRC and MOT
        c3d.export_trc(itf, trc_path, rot_mat=trf)
        c3d.export_mot(itf, mot_path, rot_mat=trf)

        # Close C3D
        ret = c3d.close_c3d(itf)

        print(f"Successfully converted C3D file to TRC and MOT formats, saved in: {output_dir}")
        return trc_path, mot_path
    except Exception as e:
        print(f"Error converting C3D file: {e}")
        return None, None


def c3d_gelenkm(rechts, src_c3d, output_dir=None):
    """
    Add elbow and hand marker points to C3D file.

    Args:
        rechts (bool): True for right side, False for left side
        src_c3d (str): Source C3D file path
        output_dir (str, optional): Output directory, defaults to processed data directory

    Returns:
        str: Modified C3D file path
    """
    try:
        print(f"[DEBUG] c3d_gelenkm called. src_c3d = {src_c3d}, rechts = {rechts}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        if output_dir is None:
            output_dir = PROCESSED_DIR

        print(f"[DEBUG] Using output directory: {output_dir}")
        ensure_dir_exists(output_dir)

        src_file_name = os.path.basename(src_c3d)
        file_name_without_ext = os.path.splitext(src_file_name)[0]
        file_ext = os.path.splitext(src_file_name)[1]

        output_file = os.path.join(output_dir, file_name_without_ext + "_gm" + file_ext)
        print(f"[DEBUG] Will write new C3D (with added markers) to: {output_file}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        dict_markers = c3d.get_dict_markers(itf, blocked_nan=True, resid=True, mask=False, desc=False, frame=False, time=False)
        dict_mkr_pos = dict_markers['DATA']['POS']

        if rechts:
            # Right side
            new_mkr_name = 'REB'
            new_mkr_pos = (dict_mkr_pos['REL'] + dict_mkr_pos['REM']) * 0.5
            c3d.add_marker(itf, new_mkr_name, new_mkr_pos, mkr_resid=None, mkr_desc='REB', log=False)

            new_mkr_hw = 'HWM'
            hw_mkr_pos = (dict_mkr_pos['RRS'] + dict_mkr_pos['RUS']) * 0.5
            c3d.add_marker(itf, new_mkr_hw, hw_mkr_pos, mkr_resid=None, mkr_desc='HWM', log=False)
        else:
            # Left side
            new_mkr_name = 'LEB'
            new_mkr_pos = (dict_mkr_pos['LEL'] + dict_mkr_pos['LEM']) * 0.5
            c3d.add_marker(itf, new_mkr_name, new_mkr_pos, mkr_resid=None, mkr_desc='LEB', log=False)

            new_mkr_hw = 'HWM'
            hw_mkr_pos = (dict_mkr_pos['LRS'] + dict_mkr_pos['LUS']) * 0.5
            c3d.add_marker(itf, new_mkr_hw, hw_mkr_pos, mkr_resid=None, mkr_desc='HWM', log=False)

        # Save modified C3D file
        ret = c3d.save_c3d(itf, output_file, compress_param_blocks=True, log=True)
        ret = c3d.close_c3d(itf, log=True)

        print(f"Successfully added marker points, saved in: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error adding marker points to C3D file: {e}")
        return None


def extract_marker_data(src_c3d):
    """
    Extract marker data from C3D file.

    Args:
        src_c3d (str): Source C3D file path

    Returns:
        dict: Marker data dictionary
    """
    try:
        print(f"[DEBUG] extract_marker_data called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        marker_names = c3d.get_marker_names(itf)
        marker_data = {}

        for marker in marker_names:
            marker_data[marker] = c3d.get_marker_data(itf, marker)

        ret = c3d.close_c3d(itf)

        return marker_data
    except Exception as e:
        print(f"Error extracting marker data: {e}")
        return None


def extract_analog_data(src_c3d):
    """
    Extract analog channel data from C3D file.

    Args:
        src_c3d (str): Source C3D file path

    Returns:
        dict: Analog channel data dictionary
    """
    try:
        print(f"[DEBUG] extract_analog_data called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        analog_names = c3d.get_analog_names(itf)
        analog_data = {}
        times = c3d.get_analog_times(itf)

        analog_data['time'] = times

        for analog in analog_names:
            analog_data[analog] = c3d.get_analog_data_unscaled(itf, analog)

        ret = c3d.close_c3d(itf)

        return analog_data
    except Exception as e:
        print(f"Error extracting analog channel data: {e}")
        return None


def get_c3d_metadata(src_c3d):
    """
    Get C3D file metadata information.

    Args:
        src_c3d (str): Source C3D file path

    Returns:
        dict: Metadata dictionary
    """
    try:
        print(f"[DEBUG] get_c3d_metadata called. src_c3d = {src_c3d}")
        print(f"[DEBUG] Does src_c3d exist on disk? {os.path.exists(src_c3d)}")

        itf = c3d.c3dserver()
        ret = c3d.open_c3d(itf, src_c3d)

        metadata = {
            # 'analog_rate': c3d.get_analog_rate(itf),
            # 'video_rate': c3d.get_video_rate(itf),
            'marker_count': len(c3d.get_marker_names(itf)),
            'analog_count': len(c3d.get_analog_names(itf)),
            'first_frame': c3d.get_first_frame(itf),
            'last_frame': c3d.get_last_frame(itf)
        }

        ret = c3d.close_c3d(itf)

        return metadata
    except Exception as e:
        print(f"Error getting C3D metadata: {e}")
        return None


def add_column_to_file(file_path, new_column_name, new_column_data):
    """
    Add new columns to file.

    Args:
        file_path (str): File path to modify
        new_column_name (list): List of column names to add
        new_column_data (list of lists): Data for each column
    """
    try:
        print(f"[DEBUG] add_column_to_file called. file_path = {file_path}")
        print(f"[DEBUG] Does file exist on disk? {os.path.exists(file_path)}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[DEBUG] Created output directory: {output_dir}")
        
        # If file doesn't exist, create a new file
        if not os.path.exists(file_path):
            print(f"[DEBUG] File doesn't exist, creating new file: {file_path}")
            with open(file_path, 'w') as file:
                # Write file header
                file.write("nRows=0\n")
                file.write("nColumns=0\n")
                file.write("endheader\n")
                file.write("\t".join(new_column_name) + "\n")
            
            # Reopen file for reading
            with open(file_path, 'r') as file:
                lines = file.readlines()
        else:
            # If file exists, read existing content
            with open(file_path, 'r') as file:
                lines = file.readlines()

        metadata_lines = []
        data_lines = []
        in_header = True

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("endheader"):
                in_header = False
                continue
            if in_header:
                metadata_lines.append(stripped)
            else:
                data_lines.append(stripped)

        metadata = {line.split('=')[0].strip(): line.split('=')[1].strip()
                    for line in metadata_lines if '=' in line}

        nRows = len(data_lines)
        max_rows = max(len(col) for col in new_column_data)

        column_headers = new_column_name

        for i, col_data in enumerate(new_column_data):
            if len(col_data) < max_rows:
                new_column_data[i].extend([None] * (max_rows - len(col_data)))

        nColumns = len(new_column_name)

        metadata["nRows"] = str(max_rows)
        metadata["nColumns"] = str(nColumns)

        fixed_file = []
        fixed_file.extend([f"{key}={value}" for key, value in metadata.items()])
        fixed_file.append("endheader")
        fixed_file.append("\t".join(column_headers))

        for i in range(max_rows):
            row_data = [str(new_column_data[j][i]) if new_column_data[j][i] is not None else ""
                        for j in range(len(new_column_data))]
            fixed_file.append("\t".join(row_data))

        with open(file_path, 'w') as file:
            file.write("\n".join(fixed_file) + "\n")

        print(f"File header has been rebuilt, new columns successfully added.")
    except Exception as e:
        print(f"Error adding columns: {e}")
        import traceback
        traceback.print_exc()


# Example usage
if __name__ == '__main__':
    # Example using relative paths
    raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'N10_n')
    processed_data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    ensure_dir_exists(processed_data_dir)

    # Example file path
    example_c3d = os.path.join(raw_data_dir, 'right', 'ROM_Ellenbogenflex_R 1.c3d')

    # Debug: Convert C3D to TRC and MOT
    print("\n===== Testing C3D to TRC and MOT conversion =====")
    trc_path, mot_path = c3dtotrc(example_c3d, processed_data_dir)

    # Check if return values are None
    if trc_path is None or mot_path is None:
        print("Conversion function returned None, an error may have occurred during conversion.")
    else:
        print(f"Conversion function return values:\n TRC path: {trc_path}\n MOT path: {mot_path}")

        # Further check if files exist on disk
        trc_exists = os.path.exists(trc_path)
        mot_exists = os.path.exists(mot_path)

        if trc_exists and mot_exists:
            print("Debug check: Both TRC and MOT files found on disk, conversion successful.")
        else:
            print("Debug check: Some files not found on disk. Please check the following status:")
            print(f"TRC file exists: {trc_exists}, MOT file exists: {mot_exists}")

    # Test add marker points functionality
    print("\n===== Testing add marker points functionality =====")
    output_c3d = c3d_gelenkm(True, example_c3d, processed_data_dir)
    if output_c3d is None:
        print("Add marker points function returned None, an error may have occurred during processing.")
    else:
        print(f"File with added marker points saved in: {output_c3d}")

        # Check if file exists on disk
        if os.path.exists(output_c3d):
            print(f"Debug check: File with added marker points found on disk, processing successful.")
        else:
            print(f"Debug check: File with added marker points not found on disk.")

    # Test extract marker data
    print("\n===== Testing extract marker data =====")
    marker_data = extract_marker_data(example_c3d)
    if marker_data is not None:
        print(f"Successfully extracted data for {len(marker_data)} markers.")
    else:
        print("Failed to extract marker data.")

    # Test extract analog channel data
    print("\n===== Testing extract analog channel data =====")
    analog_data = extract_analog_data(example_c3d)
    if analog_data is not None:
        print(f"Successfully extracted data for {len(analog_data)-1} analog channels.")  # -1 because 'time' is included
    else:
        print("Failed to extract analog channel data.")

    # Test get metadata
    print("\n===== Testing get C3D metadata =====")
    metadata = get_c3d_metadata(example_c3d)
    if metadata is not None:
        print("C3D file metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to get C3D metadata.")
