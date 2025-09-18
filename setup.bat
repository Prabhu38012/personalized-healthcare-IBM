@echo off
echo DEBUG: Starting setup process...
echo.

echo DEBUG: Current directory is:
cd
echo.

echo DEBUG: Files in current directory:
dir /b
echo.

echo DEBUG: Checking for requirements.txt...
if exist "requirements.txt" (
    echo DEBUG: requirements.txt found - OK
) else (
    echo DEBUG: requirements.txt NOT found - ERROR
    echo Make sure you're in the personalized-healthcare directory
    pause
    exit /b 1
)

echo DEBUG: Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo DEBUG: Python command failed - ERROR
    echo Please install Python and add to PATH
    pause
    exit /b 1
) else (
    echo DEBUG: Python found - OK
)

echo DEBUG: Testing venv module...
python -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo DEBUG: venv module not available - ERROR
    echo Trying alternative virtualenv...
    pip install virtualenv
    virtualenv --version
    if %errorlevel% neq 0 (
        echo DEBUG: virtualenv also failed - ERROR
        pause
        exit /b 1
    ) else (
        echo DEBUG: Using virtualenv instead of venv
        virtualenv venv
    )
) else (
    echo DEBUG: venv module available - OK
    echo DEBUG: Checking if venv folder already exists...
    if exist "venv" (
        echo DEBUG: venv folder already exists - skipping creation
    ) else (
        echo DEBUG: Creating virtual environment with python -m venv venv...
        python -m venv venv
        if %errorlevel% neq 0 (
            echo DEBUG: venv creation failed - ERROR
            pause
            exit /b 1
        ) else (
            echo DEBUG: venv creation successful - OK
        )
    )
)

echo.
echo DEBUG: Checking if venv folder exists now...
if exist "venv" (
    echo DEBUG: venv folder exists - SUCCESS
    dir venv /b
) else (
    echo DEBUG: venv folder still missing - FAILURE
    echo Something went wrong during creation
    pause
    exit /b 1
)

echo.
echo DEBUG: Attempting to activate virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo DEBUG: activate.bat found - OK
    call venv\Scripts\activate.bat
) else (
    echo DEBUG: activate.bat not found - trying alternative
    if exist "venv\Scripts\activate" (
        call venv\Scripts\activate
    ) else (
        echo DEBUG: No activation script found - ERROR
        pause
        exit /b 1
    )
)

echo.
echo DEBUG: Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo DEBUG: pip install failed - ERROR
    pause
    exit /b 1
) else (
    echo DEBUG: pip install successful - OK
)

echo.
echo DEBUG: Setup completed successfully!
echo DEBUG: venv folder should now exist
pause