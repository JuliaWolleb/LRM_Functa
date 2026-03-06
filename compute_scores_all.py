import os
import torch
import numpy as np
import csv

def extract_ssim_values(folder_path):
    ssim_values = []
    psnr_values = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)
            data = torch.load(file_path, map_location='cpu')

            if isinstance(data, dict) and 'ssim3d' in data and 'psnr' in data:
                ssim_values.append(data['ssim3d'])
                psnr_values.append(data['psnr'])
            else:
                print(f"'ssim3d' or 'psnr' not found in {filename}")

    return ssim_values, psnr_values

def extract_ssim_values_mixed(folder_path):
    ssim_values_0 = []
    psnr_values_0 = []
    ssim_values_1 = []
    psnr_values_1 = []
    ssim_values_2 = []
    psnr_values_2 = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)
            data = torch.load(file_path, map_location='cpu')

            ssim_values_0.append(data['ssim3d'])
            psnr_values_0.append(data['psnr'])
            if data.get('label') == 1:
                ssim_values_1.append(data['ssim3d'])
                psnr_values_1.append(data['psnr'])
            if data.get('label') == 2:
                ssim_values_2.append(data['ssim3d'])
                psnr_values_2.append(data['psnr'])

    return ssim_values_0, psnr_values_0, ssim_values_1, psnr_values_1, ssim_values_2, psnr_values_2

def process_folder(folder_path, mixed=False):
    if not mixed:
        ssim_list, psnr_list = extract_ssim_values(folder_path)
        ssim_array = np.array(ssim_list, dtype=np.float32)
        psnr_array = np.array(psnr_list, dtype=np.float32)
        return {
            'mean_ssim': ssim_array.mean() if len(ssim_array) > 0 else None,
            'std_ssim': ssim_array.std() if len(ssim_array) > 0 else None,
            'mean_psnr': psnr_array.mean() if len(psnr_array) > 0 else None,
            'std_psnr': psnr_array.std() if len(psnr_array) > 0 else None
        }
    else:
        ssim_list_0, psnr_list_0, ssim_list_1, psnr_list_1, ssim_list_2, psnr_list_2 = extract_ssim_values_mixed(folder_path)
        return {
            'mean_ssim0': np.array(ssim_list_0, dtype=np.float32).mean() if len(ssim_list_0) > 0 else None,
            'std_ssim0': np.array(ssim_list_0, dtype=np.float32).std() if len(ssim_list_0) > 0 else None,
            'mean_psnr0': np.array(psnr_list_0, dtype=np.float32).mean() if len(psnr_list_0) > 0 else None,
            'std_psnr0': np.array(psnr_list_0, dtype=np.float32).std() if len(psnr_list_0) > 0 else None,
            'mean_ssim1': np.array(ssim_list_1, dtype=np.float32).mean() if len(ssim_list_1) > 0 else None,
            'std_ssim1': np.array(ssim_list_1, dtype=np.float32).std() if len(ssim_list_1) > 0 else None,
            'mean_psnr1': np.array(psnr_list_1, dtype=np.float32).mean() if len(psnr_list_1) > 0 else None,
            'std_psnr1': np.array(psnr_list_1, dtype=np.float32).std() if len(psnr_list_1) > 0 else None,
            'mean_ssim2': np.array(ssim_list_2, dtype=np.float32).mean() if len(ssim_list_2) > 0 else None,
            'std_ssim2': np.array(ssim_list_2, dtype=np.float32).std() if len(ssim_list_2) > 0 else None,
            'mean_psnr2': np.array(psnr_list_2, dtype=np.float32).mean() if len(psnr_list_2) > 0 else None,
            'std_psnr2': np.array(psnr_list_2, dtype=np.float32).std() if len(psnr_list_2) > 0 else None
        }

# Main folder containing subfolders
root_folder = '/gpfs/gibbs/project/hartley/jw3234/medfuncta/vidfuncta/recontructions_newtry/'
#root_folder = '/gpfs/gibbs/project/hartley/jw3234/medfuncta/vidfuncta/reconstruct_LVH_newtry/'
output_csv = 'ssim_psnr_results_newtry.csv'
#output_csv = 'ssim_psnr_results_LVH.csv'
mixed = False  # Set True if you have labels 0,1,2

# Collect results for all subfolders (looking into nfset/test)
results = []
for subfolder in os.listdir(root_folder):

    nfset_path = os.path.join(root_folder, subfolder, 'nfset')
    test_path = os.path.join(nfset_path, 'test')
    train_path = os.path.join(nfset_path, 'train')

    
    if os.path.isdir(test_path) and len(os.listdir(test_path)) > 0:
        eval_path = test_path
        source = 'test'
    elif os.path.isdir(train_path) and len(os.listdir(train_path)) > 0:
        eval_path = train_path
        source = 'train'
    else:
        print(f"Skipping {subfolder}: no valid test or train folder")
        continue

    print('processing:', eval_path)

    stats = process_folder(eval_path, mixed=mixed)

    stats['folder'] = subfolder
    stats['source'] = source   # optional: records whether test or train was used

    results.append(stats)

print('done compute')



# Write results to CSV
if results:
    fieldnames = ['folder'] + [k for k in results[0] if k != 'folder']
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"Results saved to {output_csv}")
else:
    print("No valid nfset/test folders found.")
