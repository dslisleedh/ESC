echo 'div2k-x2' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-2.yaml --model $1 --gpu $2 &&
echo 'div2k-x3' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-3.yaml --model $1 --gpu $2 &&
echo 'div2k-x4' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-4.yaml --model $1 --gpu $2 &&

echo 'div2k-x6*' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-6.yaml --model $1 --gpu $2 &&
echo 'div2k-x12*' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-12.yaml --model $1 --gpu $2 &&
echo 'div2k-x18*' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-18.yaml --model $1 --gpu $2 &&
echo 'div2k-x24*' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-24.yaml --model $1 --gpu $2 &&
echo 'div2k-x30*' &&
PYTHONWARNINGS="ignore" python test.py --config ./configs/test/test-div2k-30.yaml --model $1 --gpu $2 &&

true
