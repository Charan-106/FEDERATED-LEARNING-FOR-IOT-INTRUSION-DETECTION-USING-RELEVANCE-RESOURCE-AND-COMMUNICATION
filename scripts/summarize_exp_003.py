import json
from pathlib import Path

path = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning\storage\experiments\exp_003\experiment_003_results.json")
with open(path) as f:
    d = json.load(f)

base_a = d['base_model_evaluation']['benchmark_a_primary']
base_b = d['base_model_evaluation']['benchmark_b_secondary']
print(f"BASE MODEL: Bench A Acc={base_a['accuracy']:.4f}%, F1={base_a['macro_f1']:.4f}%, Loss={base_a['loss']:.4f} | Bench B Acc={base_b['accuracy']:.4f}%, F1={base_b['macro_f1']:.4f}%, Loss={base_b['loss']:.4f}")
print("="*110)

for skey, sdata in d['scenarios'].items():
    sname = sdata['scenario_name']
    print(f"\n--- {sname} ({skey}) ---")
    print("Client Weights & ARC Scores:")
    for c in sdata['clients']:
        w = c['weights']
        q = c['arc_scoring']['quality_score']
        div = c['arc_scoring']['diversity_score']
        res = c['arc_scoring']['resource_score']
        comm = c['arc_scoring']['communication_score']
        print(f"  {c['name']:<35}: N={c['samples_trained']:<5} | Q={q:.1f}, D={div:.1f}, S={res:.1f}, C={comm:.1f} -> ARC={c['arc_score']:.2f} | w_FedAvg={w['fedavg']:.4f}, w_Equal={w['equal']:.4f}, w_ARC={w['arc']:.4f}")
    print("\nCandidate Evaluations on Benchmarks:")
    for ckey, cinfo in sdata['candidates'].items():
        ea = cinfo['benchmark_a_primary']
        eb = cinfo['benchmark_b_secondary']
        print(f"  {cinfo['name']:<50} | Bench A: Acc={ea['accuracy']:.4f}%, F1={ea['macro_f1']:.4f}%, Loss={ea['loss']:.4f} | Bench B: Acc={eb['accuracy']:.4f}%, F1={eb['macro_f1']:.4f}%, Loss={eb['loss']:.4f}")
