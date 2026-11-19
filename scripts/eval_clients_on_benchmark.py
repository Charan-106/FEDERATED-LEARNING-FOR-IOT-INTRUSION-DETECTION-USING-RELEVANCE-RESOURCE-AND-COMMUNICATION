import sys
from pathlib import Path
sys.path.insert(0, "ml-service")
from app.evaluation.validator import evaluate_model

p_c1 = str(Path("storage/client-updates/4b565adb-02d6-47ab-93c9-2b69b25e99d3.pt").resolve())
p_c2 = str(Path("storage/client-updates/ee3f946a-1dcb-4b28-9480-44007d803b92.pt").resolve())

res_c1 = evaluate_model(p_c1, max_samples=5000, benchmark_type="primary_capped")
print(f"Client 1 on Benchmark A: Acc={res_c1['accuracy']}%, Macro F1={res_c1['macro_f1']}%, Loss={res_c1['loss']}")

res_c2 = evaluate_model(p_c2, max_samples=5000, benchmark_type="primary_capped")
print(f"Client 2 on Benchmark A: Acc={res_c2['accuracy']}%, Macro F1={res_c2['macro_f1']}%, Loss={res_c2['loss']}")
