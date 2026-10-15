@echo off
setlocal

if /I "%~1"=="--help" goto :usage

set "JOBS=%~1"
if not defined JOBS set "JOBS=0"

pushd "%~dp0.." || exit /b 1

set "PYTHON=venv\Scripts\python.exe"
set "OUTPUT25=results\configuration_search_local"
set "OUTPUT100=results\configuration_search_local_max100"

if not exist "%PYTHON%" (
    echo ERROR: %PYTHON% does not exist.
    echo Create the virtual environment and install requirements-reproducibility.txt first.
    popd
    exit /b 2
)

set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"
set "JOBLIB_START_METHOD=loky"
set "PYTHONUNBUFFERED=1"

echo ============================================================
echo Checking the Python environment
echo ============================================================
"%PYTHON%" -m pip check
if errorlevel 1 goto :failed

echo ============================================================
echo Starting the local 1-25-term configuration search
echo Output: %OUTPUT25%
echo Workers: %JOBS% ^(0 means all available logical CPUs^)
echo ============================================================
"%PYTHON%" -m ml_meta_perf.configuration_search all ^
    --data src\ml_meta_perf\meta_dataset.csv ^
    --output "%OUTPUT25%" ^
    --jobs "%JOBS%" ^
    --min-terms 1 ^
    --plateau-tolerance 0.001 ^
    --plateau-window 3
if errorlevel 1 goto :failed

echo ============================================================
echo Starting the local 1-100-term configuration search
echo Output: %OUTPUT100%
echo Workers: %JOBS% ^(0 means all available logical CPUs^)
echo ============================================================
"%PYTHON%" -m ml_meta_perf.configuration_search all ^
    --data src\ml_meta_perf\meta_dataset.csv ^
    --output "%OUTPUT100%" ^
    --jobs "%JOBS%" ^
    --min-terms 1 ^
    --max-terms 100 ^
    --plateau-tolerance 0.001 ^
    --plateau-window 3
if errorlevel 1 goto :failed

echo ============================================================
echo Generating reports and figures
echo ============================================================
"%PYTHON%" scripts\plot_e3_valid_selection.py "%OUTPUT25%"
if errorlevel 1 goto :failed
"%PYTHON%" scripts\report_e3_valid_selection.py "%OUTPUT25%"
if errorlevel 1 goto :failed
"%PYTHON%" scripts\plot_e3_validation_curves.py "%OUTPUT25%"
if errorlevel 1 goto :failed

"%PYTHON%" scripts\plot_e3_valid_selection.py "%OUTPUT100%"
if errorlevel 1 goto :failed
"%PYTHON%" scripts\report_e3_valid_selection.py "%OUTPUT100%"
if errorlevel 1 goto :failed
"%PYTHON%" scripts\plot_e3_validation_curves.py "%OUTPUT100%"
if errorlevel 1 goto :failed

echo ============================================================
echo Both local searches completed successfully.
echo Primary results: %OUTPUT25%
echo 100-term results: %OUTPUT100%
echo ============================================================
popd
exit /b 0

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo ERROR: The local reproduction stopped with exit code %EXIT_CODE%.
echo Run this file again with the same argument to resume completed work.
popd
exit /b %EXIT_CODE%

:usage
echo Usage: scripts\equation_search_local.bat [workers]
echo.
echo workers  Number of parallel workers. The default is 0, which lets
echo          joblib use all available logical CPUs.
echo.
echo Examples:
echo   scripts\equation_search_local.bat
echo   scripts\equation_search_local.bat 8
exit /b 0
