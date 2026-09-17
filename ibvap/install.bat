@echo off
REM Installs everything IBVAP needs, then removes opencv-python-headless,
REM which easyocr pulls in as a dependency and which silently breaks
REM cv2.namedWindow (used by select_zone.py) if left installed alongside
REM the regular opencv-python. Always run this instead of a raw
REM "pip install -r requirements.txt" to avoid that trap.

pip install -r requirements.txt
pip uninstall opencv-python-headless -y

REM Force the exact known-good versions again, in case installing easyocr's
REM other dependencies (scikit-image, etc.) pulled in a newer numpy/opencv
REM behind the scenes - this has happened before and silently breaks things.
pip install --force-reinstall opencv-python==4.11.0.86 numpy==1.26.4

echo.
echo Verifying OpenCV GUI support...
python -c "import cv2; cv2.namedWindow('t'); cv2.destroyAllWindows(); print('OK - OpenCV GUI window support is working. Version:', cv2.__version__)"
