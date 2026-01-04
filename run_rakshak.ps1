$PYTHON_PATH = "C:\Users\hetbh\AppData\Local\Programs\Python\Python312\python.exe"
$STREAMLIT_PATH = "C:\Users\hetbh\AppData\Local\Programs\Python\Python312\Scripts\streamlit.exe"

Write-Host "🚀 Starting RAKSHAK AI: Indian Road Hazard Detection System..." -ForegroundColor Cyan
& $STREAMLIT_PATH run main.py
