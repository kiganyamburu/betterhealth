param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("train", "evaluate", "serve", "test")]
    [string]$Task
)

switch ($Task) {
    "train" {
        python train_model.py
    }
    "evaluate" {
        python evaluate_topk.py
    }
    "serve" {
        uvicorn api:app --reload
    }
    "test" {
        python -m pytest tests -q
    }
}
