@echo off
chcp 65001 >nul
echo ============================================================
echo  Cortex-AI -- ResNet10 runs
echo  Inicio: %date% %time%
echo ============================================================
echo.

echo [1/4] ResNet10 baseline
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run baseline
echo Fin: %time%
echo.

echo [2/4] ResNet10 age_matched
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run age_matched --age-match
echo Fin: %time%
echo.

echo [3/4] ResNet10 specaugment_freeze
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run specaugment_freeze --freeze
echo Fin: %time%
echo.

echo [4/4] ResNet10 age_matched_freeze
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run age_matched_freeze --age-match --freeze
echo Fin: %time%
echo.

echo ============================================================
echo  Todos los runs completados: %date% %time%
echo ============================================================
pause
