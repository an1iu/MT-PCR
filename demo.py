import subprocess

for i in range(0, 51, 5):
    #cmd = f"python demo_3dmatch.py --split test --benchmark 3DMatch --id {i}"
    cmd = f"python demo_vis.py --split test --benchmark 3DMatch --id {i}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
