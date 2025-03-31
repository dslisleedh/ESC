echo "ESC-Real"
echo "DRealSR"
PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/ESC_Real_X4/visualization/DRealSR -r /home2/leedh97/datasets/DRealSR/Test_x4/test_HR --device cuda
echo "RealSR"
PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/ESC_Real_X4/visualization/RealSR -r /home2/leedh97/datasets/RealSRV3/HR --device cuda
echo "RealSRSet+5images"
PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/ESC_Real_X4/visualization/RealSRSet+5images --device cuda
echo "RealLQ250"
PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/ESC_Real_X4/visualization/RealLQ250 --device cuda

# echo "SwinIR-Real"
# echo "DRealSR"
# PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/SwinIR_real_X4/visualization/DRealSR -r /home2/leedh97/datasets/DRealSR/Test_x4/test_HR --device cuda
# echo "RealSR"
# PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/SwinIR_real_X4/visualization/RealSR -r /home2/leedh97/datasets/RealSRV3/HR --device cuda
# echo "RealSRSet+5images"
# PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/SwinIR_real_X4/visualization/RealSRSet+5images --device cuda
# echo "RealLQ250"
# PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/SwinIR_real_X4/visualization/RealLQ250 --device cuda

# echo "RealESRGAN+"
# echo "DRealSR"
# PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/realesrganp/visualization/DRealSR -r /home2/leedh97/datasets/DRealSR/Test_x4/test_HR --device cuda
# echo "RealSR"
# PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/realesrganp/visualization/RealSR -r /home2/leedh97/datasets/RealSRV3/HR --device cuda
# echo "RealSRSet+5images"
# PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/realesrganp/visualization/RealSRSet+5images --device cuda
# echo "RealLQ250"
# PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/realesrganp/visualization/RealLQ250 --device cuda

# echo "DASR"
# echo "DRealSR"
# PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/DASR/visualization/DRealSR -r /home2/leedh97/datasets/DRealSR/Test_x4/test_HR --device cuda
# echo "RealSR"
# PYTHONWARNINGS="ignore" pyiqa psnry ssim lpips dists fid niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/DASR/visualization/RealSR -r /home2/leedh97/datasets/RealSRV3/HR --device cuda
# echo "RealSRSet+5images"
# PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/DASR/visualization/RealSRSet+5images --device cuda
# echo "RealLQ250"
# PYTHONWARNINGS="ignore" pyiqa niqe maniqa musiq clipiqa -t /home2/leedh97/ESC/results/DASR/visualization/RealLQ250 --device cuda