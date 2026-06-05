@echo off
chcp 65001 >nul
echo ============================================================
echo  Cortex-AI -- ResNet18 runs nocturnos
echo  Inicio: %date% %time%
echo ============================================================
echo.

echo [1/4] ResNet18 sin freeze -- 1s window
echo Inicio: %time%
uv run python -m src.models.train_resnet --run 1s_specaugment
echo Fin: %time%
echo.

echo [2/4] ResNet18 sin freeze -- 1s window age_matched
echo Inicio: %time%
uv run python -m src.models.train_resnet --run 1s_age_matched --age-match
echo Fin: %time%
echo.

echo [3/4] ResNet18 con freeze -- 1s window
echo Inicio: %time%
uv run python -m src.models.train_resnet --run 1s_specaugment_freeze --freeze
echo Fin: %time%
echo.

echo [4/4] ResNet18 con freeze -- 1s window age_matched
echo Inicio: %time%
uv run python -m src.models.train_resnet --run 1s_age_matched_freeze --age-match --freeze
echo Fin: %time%
echo.

echo [5/6] CortexCNN sin age-match -- baseline
echo Inicio: %time%
uv run python -m src.models.train_cnn --run baseline
echo Fin: %time%
echo.

echo [6/6] CortexCNN con age-match
echo Inicio: %time%
uv run python -m src.models.train_cnn --run age_matched --age-match
echo Fin: %time%
echo.

echo [7/10] ResNet10 baseline
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run baseline
echo Fin: %time%
echo.

echo [8/10] ResNet10 age_matched
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run age_matched --age-match
echo Fin: %time%
echo.

echo [9/10] ResNet10 specaugment_freeze
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run specaugment_freeze --freeze
echo Fin: %time%
echo.

echo [10/10] ResNet10 age_matched_freeze
echo Inicio: %time%
uv run python -m src.models.train_resnet10 --run age_matched_freeze --age-match --freeze
echo Fin: %time%
echo.

echo ============================================================
echo  Todos los runs completados: %date% %time%
echo ============================================================
pause
