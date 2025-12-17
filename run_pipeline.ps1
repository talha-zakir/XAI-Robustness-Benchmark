
$env:PYTHONPATH='.'

Write-Host "1. Training Baseline Model (5 epochs)..."
python src/experiments/run_train.py --epochs 5

if ($LASTEXITCODE -eq 0) {
    Write-Host "Training Complete."
} else {
    Write-Host "Training Failed."
    exit 1
}

Write-Host "2. Running Benchmark (100 samples)..."
python src/experiments/run_benchmark.py --checkpoint outputs/checkpoints/resnet18_cifar10_baseline.pth --samples 100

if ($LASTEXITCODE -eq 0) {
    Write-Host "Benchmark Complete."
} else {
    Write-Host "Benchmark Failed."
    exit 1
}

Write-Host "3. Aggregating Results..."
python src/experiments/aggregate_results.py --run_dir outputs/runs/run_001

Write-Host "4. Starting Streamlit Dashboard..."
streamlit run src/app.py
