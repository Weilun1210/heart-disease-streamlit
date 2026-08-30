$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
$LocalPython = Join-Path $Here '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $LocalPython) {
    & $LocalPython -m streamlit run "$Here\app.py"
} else {
    python -m streamlit run "$Here\app.py"
}
