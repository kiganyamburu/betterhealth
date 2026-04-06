param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("train", "evaluate", "serve", "test", "retrain")]
    [string]$Task
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot "env\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

switch ($Task) {
    "train" {
        & $PythonExe train_model.py
    }
    "evaluate" {
        & $PythonExe evaluate_topk.py
    }
    "serve" {
        & $PythonExe -m uvicorn api:app --reload
    }
    "test" {
        & $PythonExe -m pytest tests -q
    }
    "retrain" {
        # Rebuild model artifacts, then report top-k quality with the fresh model.
        & $PythonExe train_model.py
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        & $PythonExe evaluate_topk.py
    }
}
