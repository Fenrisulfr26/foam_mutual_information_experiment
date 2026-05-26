import scipy.io as sio
import numpy as np

mat_file = r'F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data\3x3 grid_scan no obj 20260514\hist_2us_100000_avg20_point05_center_obj.mat'

print("=" * 90)
print("MATLAB MAT FILE INSPECTION")
print("=" * 90)
print(f"\nFile: {mat_file}\n")

try:
    mat = sio.loadmat(mat_file)
    print("Status: File loaded successfully\n")
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

keys = sorted([k for k in mat.keys() if not k.startswith('__')])
print(f"Found {len(keys)} variables:\n")

print("VARIABLE SUMMARY:")
print("-" * 90)
print(f"{'Name':<35} {'Shape':<25} {'Dtype':<25}")
print("-" * 90)
for key in keys:
    arr = mat[key]
    print(f"{key:<35} {str(arr.shape):<25} {str(arr.dtype):<25}")

print("\n" + "=" * 90)
print("DETAILED CONTENT:\n")

for key in keys:
    arr = mat[key]
    print(f"\n[{key}]")
    print(f"  Shape: {arr.shape} | Dtype: {arr.dtype} | Elements: {arr.size}")
    
    if arr.dtype == object:
        print(f"  Content: {str(arr)[:150]}")
    elif arr.size == 0:
        print(f"  (empty)")
    else:
        if arr.ndim == 1:
            n = min(10, arr.size)
            print(f"  First {n} values: {arr[:n]}")
        elif arr.ndim == 2:
            print(f"  Matrix preview ({min(5, arr.shape[0])} x {min(5, arr.shape[1])}):")
            for i in range(min(5, arr.shape[0])):
                print(f"    {arr[i, :min(5, arr.shape[1])]}")
        else:
            print(f"  First 10 elements: {arr.flat[:10]}")

print("\n" + "=" * 90)
