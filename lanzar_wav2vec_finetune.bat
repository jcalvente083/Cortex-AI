@echo off
chcp 65001 >nul
echo ============================================================
echo  Cortex-AI -- Wav2Vec2 Finetune runs
echo  Inicio: %date% %time%
echo  Config: freeze layers 0-20, batch 4, grad_accum 4 (batch efectivo 16)
echo  Tiempo estimado: 12-38 h por run (RTX 4060)
echo ============================================================
echo.

echo [1/2] Wav2Vec2 Finetune -- baseline (con edad, todos los datos)
echo Inicio: %time%
uv run python -m src.models.train_wav2vec_finetune --run baseline --epochs 30
echo Fin: %time%
echo.

echo [2/2] Wav2Vec2 Finetune -- age_matched (sin confound de edad)
echo Inicio: %time%
uv run python -m src.models.train_wav2vec_finetune --run age_matched --age-match --epochs 30
echo Fin: %time%
echo.

echo ============================================================
echo  Todos los runs completados: %date% %time%
echo ============================================================
pause
